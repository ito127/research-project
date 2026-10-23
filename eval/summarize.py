#!/usr/bin/env python3
"""Collect <workdir>/logs/{eval,fid}_*.log into <workdir>/results.csv and <workdir>/results.md (set x mode x method)."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

LABELS = {
    "ours": "Ours (RevealLayer text removal)",
    "asuka": "ASUKA (FLUX.1-Fill)",
    "osor": "OSOR",
    "fluxfill": "FLUX.1-Fill",
    "repaint": "RePaint",
    "viteraser": "ViTEraser",
    "context": "ConText",
    "identity_input": "Input image as is (reference)",
}
# (key, label in the log, column, format, +1 = higher is better)
KEYS = [("psnr", "average psnr", "PSNR↑", "{:.2f}", 1), ("ssim", "average ssim", "MS-SSIM↑", "{:.4f}", 1),
        ("mse", "avg mse", "MSE↓", "{:.5f}", -1), ("age", "average AGE", "AGE↓", "{:.3f}", -1),
        ("peps", "average pEPS", "pEPS↓", "{:.4f}", -1), ("pceps", "average pCEPS", "pCEPS↓", "{:.4f}", -1),
        ("fid", "FID", "FID↓", "{:.2f}", -1)]
TITLES = {
    ("full813", "paste"): "All 813 test images, paste (outside the evaluated text = GT; main setting)",
    ("full813", "raw"): "All 813 test images, raw output",
    ("subset500", "paste"): "img_0..img_499, paste (includes methods run on this subset only)",
    ("subset500", "raw"): "img_0..img_499, raw output",
}


def grab(path: Path, label: str):
    if not path.is_file():
        return None
    match = re.search(rf"^{re.escape(label)}:?\s+([0-9.eE+-]+)", path.read_text(errors="ignore"), re.M)
    return float(match.group(1)) if match else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args(argv)
    workdir = args.workdir
    scope_file = workdir / "scope.txt"
    scope = scope_file.read_text().strip() if scope_file.is_file() else "?"

    rows = []
    for line in (workdir / "jobs.txt").read_text().splitlines():
        if not line.strip():
            continue
        set_, mode, method = line.split()[:3]
        tag = f"{set_}_{mode}_{method}"
        row = dict(set=set_, mode=mode, method=method)
        for key, label, *_ in KEYS:
            row[key] = grab(workdir / "logs" / (f"fid_{tag}.log" if key == "fid" else f"eval_{tag}.log"), label)
        log = workdir / "logs" / f"eval_{tag}.log"
        row["n_eval"] = len(re.findall(r"\d\s+psnr:", log.read_text(errors="ignore"))) if log.is_file() else 0
        rows.append(row)
    order = list(LABELS)
    rows.sort(key=lambda r: (r["set"], r["mode"], order.index(r["method"]) if r["method"] in order else len(order)))
    with (workdir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out = [f"# SCUT-EnsText test, {scope}", "",
           "Official ViTEraser evaluation script (PSNR / MS-SSIM / MSE / AGE / pEPS / pCEPS) and pytorch-fid. "
           "**Bold** = best per column (the reference row excluded). The official script skips images whose "
           "output equals the GT (MSE = 0); `n` is the number of evaluated images.", ""]
    for (set_, mode), title in TITLES.items():
        selected = [r for r in rows if r["set"] == set_ and r["mode"] == mode]
        if not selected:
            continue
        methods = [r for r in selected if r["method"] != "identity_input"]
        best = {key: (max if sign > 0 else min)((r[key] for r in methods if r[key] is not None), default=None)
                for key, _, _, _, sign in KEYS}
        out += [f"## {title}", "", "| Method | " + " | ".join(h for _, _, h, _, _ in KEYS) + " | n |",
                "|---" * (len(KEYS) + 2) + "|"]
        for r in selected:
            cells = []
            for key, _, _, fmt, _ in KEYS:
                value = r[key]
                text = "–" if value is None else fmt.format(value)
                bold = value is not None and r["method"] != "identity_input" and value == best[key]
                cells.append(f"**{text}**" if bold else text)
            out.append(f"| {LABELS.get(r['method'], r['method'])} | " + " | ".join(cells) + f" | {r['n_eval']} |")
        out.append("")
    (workdir / "results.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
