#!/usr/bin/env python3
"""Run OpenCV and CUDA resize benchmarks across batch sizes and plot results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OpenCV cv2.resize and cuda_resize_py pipeline by varying "
            "only batch size. Records median_ms_per_frame and plots the result."
        )
    )
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--src-width", type=int, default=3840)
    parser.add_argument("--src-height", type=int, default=1920)
    parser.add_argument("--dst-width", type=int, default=640)
    parser.add_argument("--dst-height", type=int, default=640)
    parser.add_argument("--interp", default="linear", choices=("linear", "area", "nearest", "cubic"))
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
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to test. Default: 1 2 4 8 16 32 48 64.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV, JSON, and PNG outputs.",
    )
    parser.add_argument(
        "--cuda-impl",
        default="cuda_resize_py:resize",
        help="CUDA implementation passed to bench_resize.py --impl.",
    )
    parser.add_argument(
        "--cuda-signature",
        default="src_dsize_interp",
        choices=("src", "src_dsize", "src_dsize_interp"),
    )
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help=(
            "Set bench max diff very high for CUDA runs. This does not skip the "
            "extra correctness run in bench_resize.py, but prevents diff failures."
        ),
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


def run_benchmark(args: argparse.Namespace, method: str, batch_size: int) -> dict[str, Any]:
    root = project_root()
    command = [
        sys.executable,
        str(root / "bench_resize.py"),
        "--impl",
        "opencv" if method == "opencv" else args.cuda_impl,
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

    if method == "cuda":
        command.extend(["--signature", args.cuda_signature])
        if args.skip_cuda_check:
            command.extend(["--max-diff", "1000000"])

    print(f"[run] {method:6s} batch={batch_size:<2d}")
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
        "method": method,
        "batch_size": batch_size,
        "median_ms_per_frame": data["median_ms_per_frame"],
        "median_ms_per_batch": stats["median_ms"],
        "p95_ms_per_batch": stats["p95_ms"],
        "fps_by_median": data["fps_by_median"],
        "input_throughput_GBps": data["input_throughput_GBps"],
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "batch_resize_compare.csv"
    json_path = args.out_dir / "batch_resize_compare.json"
    png_path = args.out_dir / "batch_resize_compare.png"

    fieldnames = [
        "method",
        "batch_size",
        "median_ms_per_frame",
        "median_ms_per_batch",
        "p95_ms_per_batch",
        "fps_by_median",
        "input_throughput_GBps",
    ]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "config": {
            "iters": args.iters,
            "warmup": args.warmup,
            "src_width": args.src_width,
            "src_height": args.src_height,
            "dst_width": args.dst_width,
            "dst_height": args.dst_height,
            "interp": args.interp,
            "seed": args.seed,
            "input_pool_size": args.input_pool_size,
            "batch_sizes": args.batch_sizes,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plot_results(rows, png_path)
    return csv_path, json_path, png_path


def plot_results(rows: list[dict[str, Any]], png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate the plot. Install it with:\n"
            "  conda install -c conda-forge matplotlib"
        ) from exc

    by_method: dict[str, list[dict[str, Any]]] = {"opencv": [], "cuda": []}
    for row in rows:
        by_method[row["method"]].append(row)

    plt.figure(figsize=(9, 5.2))
    for method, method_rows in by_method.items():
        method_rows.sort(key=lambda item: item["batch_size"])
        x_values = [item["batch_size"] for item in method_rows]
        y_values = [item["median_ms_per_frame"] for item in method_rows]
        label = "OpenCV cv2.resize" if method == "opencv" else "CUDA pipeline"
        marker = "o" if method == "opencv" else "s"
        plt.plot(x_values, y_values, marker=marker, linewidth=2, label=label)

    plt.xlabel("Batch size")
    plt.ylabel("Median latency per frame (ms)")
    plt.title("Resize Performance vs Batch Size")
    plt.xticks(DEFAULT_BATCH_SIZES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def print_summary(rows: list[dict[str, Any]]) -> None:
    opencv = {row["batch_size"]: row for row in rows if row["method"] == "opencv"}
    cuda = {row["batch_size"]: row for row in rows if row["method"] == "cuda"}

    print("\nSummary")
    print("batch_size, opencv_ms_per_frame, cuda_ms_per_frame, speedup")
    for batch_size in sorted(opencv):
        if batch_size not in cuda:
            continue
        opencv_ms = opencv[batch_size]["median_ms_per_frame"]
        cuda_ms = cuda[batch_size]["median_ms_per_frame"]
        speedup = opencv_ms / cuda_ms
        print(f"{batch_size}, {opencv_ms:.6f}, {cuda_ms:.6f}, {speedup:.4f}x")


def main() -> int:
    args = parse_args()
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be positive and --warmup must be non-negative")
    if args.input_pool_size <= 0:
        raise SystemExit("--input-pool-size must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise SystemExit("all --batch-sizes values must be positive")

    rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        rows.append(run_benchmark(args, "opencv", batch_size))
        rows.append(run_benchmark(args, "cuda", batch_size))

    csv_path, json_path, png_path = write_outputs(args, rows)
    print_summary(rows)
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
