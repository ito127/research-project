#!/usr/bin/env python3
"""Remove text from images with single-pass RevealLayer text removal.

One image:

    python scripts/remove_text.py --image examples/img.jpg --mask examples/mask.png --output-dir outputs/demo

A dataset (images and same-named binary masks, e.g. SCUT-EnsText test ``all_images`` / ``all_masks``):

    python scripts/remove_text.py --images-dir SCUT-EnsText/test/all_images --masks-dir SCUT-EnsText/test/all_masks \
        --scope largest_text --output-dir outputs/scut_largest_text [--shard 0 --num-shards 4]

Each sample is written to ``<output-dir>/<id>/`` (``bg_rgb.png`` is the text-free image at the RevealLayer
resolution; ``--paste-back`` also writes ``result.png`` at the input size). Samples that already have a result
are skipped, failures are logged to ``<output-dir>/failed.txt`` and the run continues.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "diffusers/src")]

from textremoval.masks import TEXT_SCOPES, load_binary_mask, select_text_mask  # noqa: E402
from textremoval.method import TextRemovalConfig, prepare_inputs, remove_text, save_result  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_file(directory: Path, stem: str) -> Path:
    matches = [directory / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES if (directory / f"{stem}{suffix}").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one file named {stem}.* in {directory}, found {matches}")
    return matches[0]


def list_samples(args) -> list[tuple[str, Path, Path]]:
    if args.image:
        return [(args.image.stem, args.image, args.mask)]
    stems = args.ids or sorted(p.stem for p in args.images_dir.iterdir() if p.suffix in IMAGE_SUFFIXES)
    samples = [(stem, find_file(args.images_dir, stem), find_file(args.masks_dir, stem)) for stem in stems]
    return samples[args.shard::args.num_shards]


def paste_back(background: Image.Image, image: Image.Image, mask: np.ndarray, geometry) -> Image.Image:
    """``background`` (RevealLayer resolution) mapped back to the input size and pasted inside the text mask."""
    content = Image.new("RGB", geometry.content_size)
    content.paste(background, geometry.offset)
    restored = np.asarray(content.resize(image.size, Image.Resampling.LANCZOS))
    return Image.fromarray(np.where(mask[..., None], restored, np.asarray(image.convert("RGB"))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("input (one image, or a directory pair)")
    source.add_argument("--image", type=Path)
    source.add_argument("--mask", type=Path, help="Binary text mask (white = text), same size as --image.")
    source.add_argument("--images-dir", type=Path)
    source.add_argument("--masks-dir", type=Path, help="Masks named like the images (any image suffix).")
    source.add_argument("--ids", nargs="*", help="Sample ids (file stems) to run; default: every image.")
    parser.add_argument("--scope", choices=TEXT_SCOPES, default="largest_text",
                        help="largest_text: largest 8-connected component of the mask; all_text: the whole mask.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--paste-back", action="store_true",
                        help="Also write result.png: input size, the input image outside the text mask.")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Also write the prepared inputs, per-step masks, a step log and clean-estimate previews.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--gpu-id", type=int, default=0)
    defaults = TextRemovalConfig()
    method = parser.add_argument_group("method (defaults = the reported setting)")
    for name, value in vars(defaults).items():
        method.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.image) == bool(args.images_dir) or bool(args.image) != bool(args.mask) \
            or bool(args.images_dir) != bool(args.masks_dir):
        raise SystemExit("give either --image and --mask, or --images-dir and --masks-dir")
    config = TextRemovalConfig(**{name: getattr(args, name) for name in vars(TextRemovalConfig())})
    samples = list_samples(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from textremoval.runtime import load_models

    pipeline, decoder, device = load_models(args.models_dir, gpu_id=args.gpu_id, max_layer=config.max_layer)
    failed = []
    for index, (stem, image_path, mask_path) in enumerate(samples, 1):
        out_dir = args.output_dir / stem
        if (out_dir / "metadata.json").is_file() and not args.overwrite:
            print(f"[{index}/{len(samples)}] {stem}: exists, skipped")
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            text_mask = select_text_mask(load_binary_mask(mask_path, args.mask_threshold), args.scope)
            inputs = prepare_inputs(image, text_mask, config)
            result = remove_text(pipeline, decoder, device, inputs, config, record_previews=args.save_intermediates)
            save_result(out_dir, pipeline, inputs, result, config, save_intermediates=args.save_intermediates,
                        extra_metadata={"sample": stem, "image": str(image_path), "mask": str(mask_path),
                                        "scope": args.scope})
            if args.paste_back:
                paste_back(result[0], image, text_mask, inputs.geometry).save(out_dir / "result.png")
            print(f"[{index}/{len(samples)}] {stem}: {result[-1]:.1f} s")
        except Exception as error:  # keep going; the failure is reported at the end
            failed.append(stem)
            print(f"[{index}/{len(samples)}] {stem}: FAILED ({error})", file=sys.stderr)
            traceback.print_exc()
            with (args.output_dir / "failed.txt").open("a", encoding="utf-8") as handle:
                handle.write(f"{stem}\t{type(error).__name__}: {error}\n")
    print(f"done: {len(samples) - len(failed)} ok, {len(failed)} failed {failed if failed else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
