#!/usr/bin/env python3
"""Plot resize benchmark CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_ORDER = [
    "baseline",
    "cpu_float",
    "cpu_fixed",
    "cpu_simd",
    "cuda",
    "cuda_resize_only",
]

STYLES = {
    "baseline": ("OpenCV baseline", "o", "-"),
    "cpu_float": ("CPU float", "D", "-"),
    "cpu_fixed": ("CPU fixed-point", "s", "-"),
    "cpu_simd": ("CPU SIMD", "P", "-"),
    "cuda": ("CUDA end-to-end", "^", "-"),
    "cuda_resize_only": ("CUDA resize only", "x", "--"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read resize CSV results and generate a PNG plot.")
    parser.add_argument("--csv", type=Path, default=Path("results_cpu/resize_suite.csv"))
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--title", default="Resize Performance vs Batch Size")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append({
                "method": row["method"],
                "batch_size": int(row["batch_size"]),
                "median_ms_per_frame": float(row["median_ms_per_frame"]),
            })
    return rows


def plot_rows(rows: list[dict[str, Any]], png_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    methods = [method for method in DEFAULT_ORDER if any(row["method"] == method for row in rows)]
    batch_sizes = sorted({row["batch_size"] for row in rows})

    plt.figure(figsize=(10.5, 5.8))
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: row["batch_size"])
        label, marker, linestyle = STYLES[method]
        plt.plot(
            [row["batch_size"] for row in method_rows],
            [row["median_ms_per_frame"] for row in method_rows],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    plt.xlabel("Batch size")
    plt.ylabel("Median latency per frame (ms)")
    plt.title(title)
    plt.xticks(batch_sizes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    png_path = args.png if args.png is not None else args.csv.with_suffix(".png")
    rows = load_rows(args.csv)
    plot_rows(rows, png_path, args.title)
    print(f"Wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
