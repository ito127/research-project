#!/usr/bin/env python3
"""Per-image scores and a qualitative comparison figure of the cases where our method works best.

Score = PSNR inside the evaluation mask L of the paste target against the GT (outside L every method equals the GT,
so this is the per-image quantity the paste table averages). Among masks with at least ``--min-area`` pixels (so the
region is visible), the figure shows the ``--cases`` images with the largest PSNR_ours - max(PSNR of the other
methods): a crop around L of the input (L outlined in red), the GT and every method.

Uses ``targets/<set>/paste/<method>`` of ``build_targets.py`` (``--set subset500`` to include methods run on
img_0..img_499 only). Writes per_image_psnr.csv, qualitative.png and qualitative_cases.csv into the workdir.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from summarize import LABELS

CELL = 220
GAP = 4


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) >= 128


def masked_psnr(out: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    mse = ((out[mask].astype(np.float64) - gt[mask].astype(np.float64)) / 255.0) ** 2
    return 10 * np.log10(1.0 / max(mse.mean(), 1e-10))


def crop_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(min(max(max(np.ptp(ys), np.ptp(xs)) * 1.6 + 32, 128), min(mask.shape)))
    y0 = int(np.clip(cy - side / 2, 0, mask.shape[0] - side))
    x0 = int(np.clip(cx - side / 2, 0, mask.shape[1] - side))
    return x0, y0, x0 + side, y0 + side


def outline(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    edge = m.copy()
    edge[1:-1, 1:-1] &= ~inner
    return edge


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def scores(targets: Path, methods: list[str], out_csv: Path) -> list[dict]:
    rows = []
    for gt_path in sorted((targets / "gt").glob("*.png")):
        stem = gt_path.stem
        mask = load_mask(targets / "masks" / f"{stem}.png")
        if not mask.any():
            continue
        gt = rgb(gt_path)
        row = dict(id=stem, mask_area=int(mask.sum()))
        for method in methods:
            row[method] = masked_psnr(rgb(targets / "paste" / method / f"{stem}.png"), gt, mask)
        others = np.array([row[m] for m in methods[1:]])
        row["ours_minus_best_other"] = row[methods[0]] - others.max()
        row["ours_rank"] = int(1 + (others > row[methods[0]]).sum())
        rows.append(row)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    return rows


def figure(cases, methods, targets: Path, images_dir: Path, path: Path) -> None:
    """Rows = cases; columns = input crop (text outlined), GT, then every method (ours first, highlighted)."""
    regular = font(17)
    columns = ["Input", "GT"] + [LABELS.get(m, m).split(" (")[0] for m in methods]
    head = 34
    canvas = Image.new("RGB", (CELL * len(columns), head + (CELL + GAP) * len(cases) - GAP), "white")
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(columns):
        x = c * CELL
        if c == 2:
            draw.rectangle([x, 0, x + CELL - 1, head - 2], fill=(255, 236, 200))
        draw.text((x + (CELL - draw.textlength(name, font=regular)) / 2, 6), name, font=regular, fill="black")
    for r, case in enumerate(cases):
        stem = case["id"]
        y = head + r * (CELL + GAP)
        mask = load_mask(targets / "masks" / f"{stem}.png")
        box = crop_box(mask)
        marked = rgb(images_dir / f"{stem}.jpg").copy()
        marked[outline(mask)] = (255, 0, 0)
        cells = [Image.fromarray(marked), Image.open(targets / "gt" / f"{stem}.png").convert("RGB")]
        cells += [Image.open(targets / "paste" / m / f"{stem}.png").convert("RGB") for m in methods]
        for c, cell in enumerate(cells):
            canvas.paste(cell.crop(box).resize((CELL - GAP, CELL - GAP), Image.LANCZOS), (c * CELL + GAP // 2, y))
    canvas.save(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True, help="SCUT-EnsText test all_images/.")
    parser.add_argument("--set", choices=("full813", "subset500"), default="full813")
    parser.add_argument("--methods", nargs="+", required=True, help="Methods to show; the first one is ours.")
    parser.add_argument("--min-area", type=int, default=1500)
    parser.add_argument("--cases", type=int, default=5)
    args = parser.parse_args(argv)
    targets = args.workdir / "targets" / args.set
    rows = scores(targets, args.methods, args.workdir / "per_image_psnr.csv")
    pool = [r for r in rows if r["mask_area"] >= args.min_area]
    cases = sorted(pool, key=lambda r: -r["ours_minus_best_other"])[:args.cases]
    figure(cases, args.methods, targets, args.images_dir, args.workdir / "qualitative.png")
    with (args.workdir / "qualitative_cases.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "mask_area", "ours_rank"] + args.methods + ["ours_minus_best_other"])
        for c in cases:
            writer.writerow([c["id"], c["mask_area"], c["ours_rank"]] + [f"{c[m]:.2f}" for m in args.methods]
                            + [f"{c['ours_minus_best_other']:.2f}"])
    print(f"{len(rows)} images scored, {len(pool)} with mask >= {args.min_area}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
