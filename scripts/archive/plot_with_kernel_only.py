#!/usr/bin/env python3
"""Add a CUDA kernel-only line to the existing optimized comparison plot."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cuda_resize_py


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot OpenCV, CUDA before, CUDA optimized, and CUDA kernel-only lines."
    )
    parser.add_argument("--input-csv", type=Path, default=Path("results/batch_resize_compare_optimized.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--src-width", type=int, default=3840)
    parser.add_argument("--src-height", type=int, default=1920)
    parser.add_argument("--dst-width", type=int, default=640)
    parser.add_argument("--dst-height", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def load_rows(path: Path, batch_sizes: list[int]) -> list[dict[str, Any]]:
    wanted = set(batch_sizes)
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            batch_size = int(row["batch_size"])
            if batch_size not in wanted:
                continue
            rows.append({
                "method": row["method"],
                "batch_size": batch_size,
                "median_ms_per_frame": float(row["median_ms_per_frame"]),
                "median_ms_per_batch": float(row["median_ms_per_batch"]),
                "p95_ms_per_batch": float(row["p95_ms_per_batch"]),
                "fps_by_median": float(row["fps_by_median"]),
                "input_throughput_GBps": float(row["input_throughput_GBps"]),
            })
    return rows


def make_input(args: argparse.Namespace, batch_size: int) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    if batch_size == 1:
        shape = (args.src_height, args.src_width, 3)
    else:
        shape = (batch_size, args.src_height, args.src_width, 3)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def profile_kernel_only(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    src = make_input(args, batch_size)
    dsize = (args.dst_width, args.dst_height)

    for _ in range(args.warmup):
        cuda_resize_py.resize(src, dsize, cv2.INTER_LINEAR)

    kernel_ms_per_frame: list[float] = []
    for _ in range(args.iters):
        _, profile = cuda_resize_py.resize_profile(src, dsize, cv2.INTER_LINEAR)
        kernel_ms_per_frame.append(float(profile["kernel_ms_per_frame"]))

    median = statistics.median(kernel_ms_per_frame)
    fps = 1000.0 / median
    print(f"[profile] cuda_kernel_only batch={batch_size:<2d} median={median:.6f} ms/frame")
    return {
        "method": "cuda_kernel_only",
        "batch_size": batch_size,
        "median_ms_per_frame": median,
        "median_ms_per_batch": median * batch_size,
        "p95_ms_per_batch": percentile(kernel_ms_per_frame, 0.95) * batch_size,
        "fps_by_median": fps,
        "input_throughput_GBps": args.src_width * args.src_height * 3 * fps / 1_000_000_000.0,
    }


def plot_rows(rows: list[dict[str, Any]], png_path: Path, batch_sizes: list[int]) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "opencv": ("OpenCV baseline", "o", "-"),
        "cuda_before": ("CUDA before", "s", "-"),
        "cuda_optimized": ("CUDA optimized", "^", "-"),
        "cuda_kernel_only": ("CUDA kernel only, no transfer", "x", "--"),
    }

    plt.figure(figsize=(10, 5.6))
    for method in ("opencv", "cuda_before", "cuda_optimized", "cuda_kernel_only"):
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda item: item["batch_size"])
        x_values = [row["batch_size"] for row in method_rows]
        y_values = [row["median_ms_per_frame"] for row in method_rows]
        label, marker, linestyle = styles[method]
        plt.plot(x_values, y_values, marker=marker, linestyle=linestyle, linewidth=2, label=label)

    plt.xlabel("Batch size")
    plt.ylabel("Median latency per frame (ms)")
    plt.title("Resize Performance vs Batch Size")
    plt.xticks(batch_sizes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "batch_resize_compare_with_kernel.csv"
    json_path = args.out_dir / "batch_resize_compare_with_kernel.json"
    png_path = args.out_dir / "batch_resize_compare_with_kernel.png"

    fieldnames = [
        "method",
        "batch_size",
        "median_ms_per_frame",
        "median_ms_per_batch",
        "p95_ms_per_batch",
        "fps_by_median",
        "input_throughput_GBps",
    ]
    rows = sorted(rows, key=lambda row: (row["batch_size"], row["method"]))
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps({
            "config": {
                "input_csv": str(args.input_csv),
                "batch_sizes": args.batch_sizes,
                "warmup": args.warmup,
                "iters": args.iters,
                "kernel_only_definition": "cuda event elapsed kernel_ms_per_frame from resize_profile",
            },
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )
    plot_rows(rows, png_path, args.batch_sizes)
    return csv_path, json_path, png_path


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"input CSV not found: {args.input_csv}")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--iters must be positive and --warmup must be non-negative")

    rows = load_rows(args.input_csv, args.batch_sizes)
    for batch_size in args.batch_sizes:
        rows.append(profile_kernel_only(args, batch_size))

    csv_path, json_path, png_path = write_outputs(args, rows)
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
