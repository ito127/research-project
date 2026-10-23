#!/usr/bin/env python3
"""Per-image scores and qualitative comparison figures (cases where our method succeeds / fails).

Score = PSNR inside the evaluation mask L of the paste target against the GT (outside L every method equals the GT,
so this is the per-image quantity the paste table averages). Only masks with at least ``--min-area`` pixels are
selected so the region is visible:
  success : largest  PSNR_ours - max(PSNR of the other methods)
  failure : smallest PSNR_ours - median(PSNR of the other methods)
Uses ``targets/<set>/paste/<method>`` of ``build_targets.py`` (``--set subset500`` to include methods run on
img_0..img_499 only). Writes per_image_psnr.csv, qualitative_success.png, qualitative_failure.png and
qualitative_cases.csv into the workdir.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from summarize import LABELS

CELL = 220


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


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
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def scores(targets: Path, methods: list[str], out_csv: Path) -> list[dict]:
    rows = []
    for gt_path in sorted((targets / "gt").glob("*.png")):
        stem = gt_path.stem
        mask = np.asarray(Image.open(targets / "masks" / f"{stem}.png").convert("L")) >= 128
        if not mask.any():
            continue
        gt = rgb(gt_path)
        row = dict(id=stem, mask_area=int(mask.sum()))
        for method in methods:
            row[method] = masked_psnr(rgb(targets / "paste" / method / f"{stem}.png"), gt, mask)
        others = np.array([row[m] for m in methods[1:]])
        row["ours_minus_best_other"] = row[methods[0]] - others.max()
        row["ours_minus_median_other"] = row[methods[0]] - np.median(others)
        row["ours_rank"] = int(1 + (others > row[methods[0]]).sum())
        rows.append(row)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    return rows


def figure(cases, methods, targets: Path, images_dir: Path, title: str, path: Path) -> None:
    regular, small, big = font(17), font(15), font(22)
    columns = ["Input", "Input (zoom)", "GT"] + [LABELS.get(m, m).split(" (")[0] for m in methods]
    left, head, foot, top = 90, 34, 24, 70
    canvas = Image.new("RGB", (left + CELL * len(columns), top + head + (CELL + foot) * len(cases)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), title, font=big, fill="black")
    draw.text((10, 40), "Below each cell: PSNR inside the evaluated text (red = best in the row). "
                        "Left: image id / mask area / ours minus best other.", font=small, fill=(60, 60, 60))
    for c, name in enumerate(columns):
        x = left + c * CELL
        if c == 3:
            draw.rectangle([x, top, x + CELL - 1, top + head - 2], fill=(255, 236, 200))
        draw.text((x + (CELL - draw.textlength(name, font=regular)) / 2, top + 6), name, font=regular, fill="black")
    for r, case in enumerate(cases):
        stem = case["id"]
        y = top + head + r * (CELL + foot)
        mask = np.asarray(Image.open(targets / "masks" / f"{stem}.png").convert("L")) >= 128
        box = crop_box(mask)
        image = rgb(images_dir / f"{stem}.jpg")
        full = Image.fromarray(image)
        ImageDraw.Draw(full).rectangle(box, outline=(255, 0, 0), width=4)
        marked = image.copy()
        marked[outline(mask)] = (255, 0, 0)
        cells = [full.resize((CELL, CELL), Image.LANCZOS),
                 Image.fromarray(marked).crop(box).resize((CELL, CELL), Image.LANCZOS),
                 Image.open(targets / "gt" / f"{stem}.png").convert("RGB").crop(box).resize((CELL, CELL), Image.LANCZOS)]
        cells += [Image.open(targets / "paste" / m / f"{stem}.png").convert("RGB").crop(box)
                  .resize((CELL, CELL), Image.LANCZOS) for m in methods]
        best = max(methods, key=lambda m: case[m])
        for c, cell in enumerate(cells):
            x = left + c * CELL
            canvas.paste(cell, (x + 2, y + 2))
            if c >= 3:
                method = methods[c - 3]
                text = f"{case[method]:.2f} dB"
                draw.text((x + (CELL - draw.textlength(text, font=small)) / 2, y + CELL + 2), text, font=small,
                          fill=(200, 0, 0) if method == best else "black")
        draw.text((8, y + CELL / 2 - 30), stem, font=regular, fill="black")
        draw.text((8, y + CELL / 2 - 6), f"{case['mask_area']}px", font=small, fill=(90, 90, 90))
        draw.text((8, y + CELL / 2 + 14), f"Δ{case['ours_minus_best_other']:+.1f}", font=small, fill=(90, 90, 90))
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
    success = sorted(pool, key=lambda r: -r["ours_minus_best_other"])[:args.cases]
    failure = sorted(pool, key=lambda r: r["ours_minus_median_other"])[:args.cases]
    others = len(args.methods) - 1
    figure(success, args.methods, targets, args.images_dir,
           f"Success cases (largest ours - best of {others} other methods)", args.workdir / "qualitative_success.png")
    figure(failure, args.methods, targets, args.images_dir,
           f"Failure cases (smallest ours - median of {others} other methods)", args.workdir / "qualitative_failure.png")
    with (args.workdir / "qualitative_cases.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "id", "mask_area", "ours_rank"] + args.methods
                        + ["ours_minus_best_other", "ours_minus_median_other"])
        for kind, cases in (("success", success), ("failure", failure)):
            for c in cases:
                writer.writerow([kind, c["id"], c["mask_area"], c["ours_rank"]] + [f"{c[m]:.2f}" for m in args.methods]
                                + [f"{c['ours_minus_best_other']:.2f}", f"{c['ours_minus_median_other']:.2f}"])
    print(f"{len(rows)} images scored, {len(pool)} with mask >= {args.min_area}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
