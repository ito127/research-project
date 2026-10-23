#!/usr/bin/env python3
"""Build the SCUT-EnsText evaluation targets for our method and other text-removal methods.

Protocol (``--scope largest_text``, the main setting): L = the largest 8-connected component of the SCUT text mask
(first in scan order on ties), GT = the official text-removed image inside L and the input image outside L, all at
the native 512x512 size, no mask dilation. ``--scope all_text``: L = the whole SCUT mask, GT = the official GT.

Two targets per method:
  raw   : the method's output as is
  paste : the method's output inside L, GT outside L (compares only the removal inside L)
Methods that erase every text in the image (e.g. ViTEraser, ConText) have no meaningful largest_text raw target;
list them with ``--paste-only``.

Two evaluation sets: ``full813`` (every test image) and, for largest_text, ``subset500`` (img_0..img_499) for
methods that were only run on that subset (``--subset-only``). Writes
``<workdir>/targets/<set>/<mode>/<method>/<id>.png``, ``<workdir>/targets/<set>/{gt,masks}/<id>.png`` and
``<workdir>/jobs.txt`` (``<set> <mode> <method> <target dir> <gt dir>`` per line) for ``run_eval.sh``.

Example (largest_text, our results from ``scripts/remove_text.py --scope largest_text``):

    python eval/build_targets.py --scut-root data/SCUT-EnsText/test --scope largest_text --workdir eval_runs/largest \
        --ours outputs/scut_largest_text \
        --method osor='baselines/osor/{id}.png' --method viteraser='baselines/viteraser/{id}.png' \
        --method repaint='baselines/repaint/{num:04d}_layer_0.png' --paste-only viteraser --subset-only repaint
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textremoval.masks import select_text_mask  # noqa: E402

SUBSET = [f"img_{i}" for i in range(500)]


def parse_method(value: str) -> tuple[str, str]:
    name, sep, pattern = value.partition("=")
    if not sep or not name or not pattern:
        raise argparse.ArgumentTypeError("--method must be NAME=PATTERN, e.g. osor='out/{id}.png'")
    return name, pattern


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scut-root", type=Path, required=True,
                        help="SCUT-EnsText test folder with all_images/, all_gts/ and all_masks/.")
    parser.add_argument("--scope", choices=("largest_text", "all_text"), default="largest_text")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--ours", type=Path, help="Output directory of scripts/remove_text.py.")
    parser.add_argument("--ours-pattern", default="{id}/bg_rgb.png",
                        help="Result path inside --ours ({id} = image id). Resized to the native size (LANCZOS).")
    parser.add_argument("--method", type=parse_method, action="append", default=[],
                        help="Another method as NAME=PATTERN; {id} = image id, {num} = its number (img_12 -> 12).")
    parser.add_argument("--paste-only", nargs="*", default=[], help="Methods without a raw target.")
    parser.add_argument("--subset-only", nargs="*", default=[],
                        help="Methods run on img_0..img_499 only (largest_text: evaluated on subset500).")
    parser.add_argument("--no-identity", action="store_true", help="Skip the reference row 'input image as is'.")
    return parser


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def link_dir(dst: Path, src: Path, stems) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        link = dst / f"{stem}.png"
        if not link.exists():
            os.symlink((src / f"{stem}.png").resolve(), link)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    targets = args.workdir / "targets"
    stems = sorted(p.stem for p in (args.scut_root / "all_images").glob("*.jpg"))
    subset = [s for s in SUBSET if s in stems] if args.scope == "largest_text" else []

    # GT and evaluation masks.
    masks = {}
    for stem in stems:
        image = load_rgb(args.scut_root / "all_images" / f"{stem}.jpg")
        full = np.asarray(Image.open(args.scut_root / "all_masks" / f"{stem}.png").convert("L")) >= 128
        mask = select_text_mask(full, args.scope) if full.any() else full
        gt = load_rgb(args.scut_root / "all_gts" / f"{stem}.jpg")
        masks[stem] = mask
        save(np.where(mask[..., None], gt, image) if args.scope == "largest_text" else gt,
             targets / "full813" / "gt" / f"{stem}.png")
        save(mask.astype(np.uint8) * 255, targets / "full813" / "masks" / f"{stem}.png")
    if subset:
        link_dir(targets / "subset500" / "gt", targets / "full813" / "gt", subset)
        link_dir(targets / "subset500" / "masks", targets / "full813" / "masks", subset)

    methods = ([("ours", None)] if args.ours else []) + list(args.method) \
        + ([] if args.no_identity else [("identity_input", None)])
    jobs = []
    for name, pattern in methods:
        only_subset = name in args.subset_only and subset
        set_ = "subset500" if only_subset else "full813"
        modes = ["paste"] if name in args.paste_only else ["raw", "paste"]
        for stem in subset if only_subset else stems:
            image = load_rgb(args.scut_root / "all_images" / f"{stem}.jpg")
            gt = load_rgb(targets / "full813" / "gt" / f"{stem}.png")
            if name == "identity_input":
                output = image
            elif name == "ours":
                path = args.ours / args.ours_pattern.format(id=stem)
                if not path.is_file():
                    # Only an empty text mask has no result (nothing to remove): the output is the input.
                    if masks[stem].any():
                        raise FileNotFoundError(f"missing result of our method: {path}")
                    output = image
                else:
                    output = np.asarray(Image.open(path).convert("RGB").resize(image.shape[1::-1], Image.LANCZOS))
            else:
                output = load_rgb(Path(pattern.format(id=stem, num=int(stem.split("_")[-1]))))
            if output.shape != gt.shape:
                raise ValueError(f"{name} {stem}: output {output.shape} does not match GT {gt.shape}")
            if "raw" in modes:
                save(output, targets / set_ / "raw" / name / f"{stem}.png")
            save(np.where(masks[stem][..., None], output, gt), targets / set_ / "paste" / name / f"{stem}.png")
        for mode in modes:
            jobs.append((set_, mode, name, targets / set_ / mode / name, targets / set_ / "gt"))
            if set_ == "full813" and subset:
                link_dir(targets / "subset500" / mode / name, targets / "full813" / mode / name, subset)
                jobs.append(("subset500", mode, name, targets / "subset500" / mode / name, targets / "subset500" / "gt"))
        print(name, modes, set_, flush=True)
    (args.workdir / "jobs.txt").write_text("".join(" ".join(map(str, job)) + "\n" for job in jobs))
    (args.workdir / "scope.txt").write_text(args.scope + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
