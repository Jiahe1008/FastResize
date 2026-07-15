#!/usr/bin/env python3
"""Rerun only OpenCV baseline rows in results_cpu and regenerate the plot."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]
FIELDNAMES = [
    "method",
    "batch_size",
    "median_ms_per_frame",
    "median_ms_per_batch",
    "p95_ms_per_batch",
    "fps_by_median",
    "input_throughput_GBps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update only OpenCV baseline rows in results_cpu CSV/JSON."
    )
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--input-pool-size",
        type=int,
        default=1,
        help=(
            "Number of pre-generated inputs bench_resize.py rotates through. "
            "Default 1 preserves the original repeated-input benchmark."
        ),
    )
    parser.add_argument("--src-width", type=int, default=3840)
    parser.add_argument("--src-height", type=int, default=1920)
    parser.add_argument("--dst-width", type=int, default=640)
    parser.add_argument("--dst-height", type=int, default=640)
    parser.add_argument(
        "--interp",
        choices=("linear", "area", "nearest", "cubic"),
        default="linear",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--out-dir", type=Path, default=Path("results_cpu"))
    parser.add_argument(
        "--basename",
        default="batch_resize_cpu_cuda_baseline",
        help="Output stem under --out-dir.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def extract_json(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"benchmark did not print JSON:\n{stdout}")
    return json.loads(stdout[start : end + 1])


def run_opencv(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    root = project_root()
    command = [
        sys.executable,
        str(root / "bench_resize.py"),
        "--impl",
        "opencv",
        "--interp",
        args.interp,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--batch-size",
        str(batch_size),
        "--src-width",
        str(args.src_width),
        "--src-height",
        str(args.src_height),
        "--dst-width",
        str(args.dst_width),
        "--dst-height",
        str(args.dst_height),
        "--seed",
        str(args.seed),
        "--input-pool-size",
        str(args.input_pool_size),
        "--json",
    ]
    print(f"[run] opencv batch={batch_size:<2d}")
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "benchmark failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    data = extract_json(completed.stdout)
    stats = data["stats"]
    return {
        "method": "opencv",
        "batch_size": batch_size,
        "median_ms_per_frame": data["median_ms_per_frame"],
        "median_ms_per_batch": stats["median_ms"],
        "p95_ms_per_batch": stats["p95_ms"],
        "fps_by_median": data["fps_by_median"],
        "input_throughput_GBps": data["input_throughput_GBps"],
    }


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            rows.append({
                "method": row["method"],
                "batch_size": int(row["batch_size"]),
                "median_ms_per_frame": float(row["median_ms_per_frame"]),
                "median_ms_per_batch": float(row["median_ms_per_batch"]),
                "p95_ms_per_batch": float(row["p95_ms_per_batch"]),
                "fps_by_median": float(row["fps_by_median"]),
                "input_throughput_GBps": float(row["input_throughput_GBps"]),
            })
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, Any]], png_path: Path, batch_sizes: list[int]) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "opencv": ("OpenCV baseline", "o", "-"),
        "cpu": ("CPU multithread", "D", "-"),
        "cuda": ("CUDA end-to-end", "^", "-"),
        "cuda_kernel_only": ("CUDA resize only, no transfer", "x", "--"),
    }

    plt.figure(figsize=(10, 5.6))
    for method in ("opencv", "cpu", "cuda", "cuda_kernel_only"):
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda item: item["batch_size"])
        label, marker, linestyle = styles[method]
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
    plt.title("Resize Performance vs Batch Size")
    plt.xticks(batch_sizes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def print_summary(rows: list[dict[str, Any]], batch_sizes: list[int]) -> None:
    by_method_batch = {
        (row["method"], row["batch_size"]): row for row in rows
    }
    print("\nUpdated OpenCV baseline")
    print("batch, opencv_ms_per_frame")
    for batch_size in batch_sizes:
        row = by_method_batch[("opencv", batch_size)]
        print(f"{batch_size}, {row['median_ms_per_frame']:.6f}")


def main() -> int:
    args = parse_args()
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be positive and --warmup must be non-negative")
    if args.input_pool_size <= 0:
        raise SystemExit("--input-pool-size must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise SystemExit("all batch sizes must be positive")

    root = project_root()
    csv_path = root / args.out_dir / f"{args.basename}.csv"
    json_path = root / args.out_dir / f"{args.basename}.json"
    png_path = root / args.out_dir / f"{args.basename}.png"
    if not csv_path.exists() or not json_path.exists():
        raise SystemExit(f"missing existing result files under {root / args.out_dir}")

    rows = read_csv_rows(csv_path)
    baseline_rows = {
        row["batch_size"]: row
        for row in (run_opencv(args, batch_size) for batch_size in args.batch_sizes)
    }

    updated_rows: list[dict[str, Any]] = []
    seen_baseline_batches: set[int] = set()
    for row in rows:
        if row["method"] == "opencv" and row["batch_size"] in baseline_rows:
            updated_rows.append(baseline_rows[row["batch_size"]])
            seen_baseline_batches.add(row["batch_size"])
        else:
            updated_rows.append(row)

    missing = set(args.batch_sizes) - seen_baseline_batches
    if missing:
        raise SystemExit(f"CSV did not contain OpenCV rows for batches: {sorted(missing)}")

    write_csv_rows(csv_path, updated_rows)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["rows"] = updated_rows
    data.setdefault("config", {})
    data["config"].update({
        "baseline_batch_preallocated": True,
        "baseline_input_pool_size": args.input_pool_size,
        "baseline_rerun_iters": args.iters,
        "baseline_rerun_warmup": args.warmup,
    })
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    plot_rows(updated_rows, png_path, args.batch_sizes)
    print_summary(updated_rows, args.batch_sizes)
    print(f"\nUpdated: {csv_path}")
    print(f"Updated: {json_path}")
    print(f"Updated: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
