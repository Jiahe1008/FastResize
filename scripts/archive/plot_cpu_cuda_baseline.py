#!/usr/bin/env python3
"""Plot OpenCV baseline, CPU resize, CUDA end-to-end, and CUDA kernel-only."""

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
            "Create a four-line resize comparison plot: baseline, CPU, "
            "current CUDA, and CUDA resize-only kernel."
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
            "Number of pre-generated inputs bench_resize.py rotates through "
            "for CPU runs. Default 1 preserves the original benchmark."
        ),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument(
        "--cuda-csv",
        type=Path,
        default=Path("cuda_results/batch_resize_compare_with_kernel.csv"),
        help="CSV containing opencv, cuda_optimized, and cuda_kernel_only rows.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results_cpu"))
    parser.add_argument("--cpu-impl", default="fast_cpu_resize:resize")
    parser.add_argument(
        "--cpu-signature",
        default="src_dsize_interp",
        choices=("src", "src_dsize", "src_dsize_interp"),
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


def load_cuda_rows(path: Path, batch_sizes: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted = set(batch_sizes)
    method_map = {
        "opencv": "opencv",
        "cuda_optimized": "cuda",
        "cuda_kernel_only": "cuda_kernel_only",
    }

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            batch_size = int(row["batch_size"])
            if batch_size not in wanted:
                continue
            method = method_map.get(row["method"])
            if method is None:
                continue
            rows.append({
                "method": method,
                "batch_size": batch_size,
                "median_ms_per_frame": float(row["median_ms_per_frame"]),
                "median_ms_per_batch": float(row["median_ms_per_batch"]),
                "p95_ms_per_batch": float(row["p95_ms_per_batch"]),
                "fps_by_median": float(row["fps_by_median"]),
                "input_throughput_GBps": float(row["input_throughput_GBps"]),
            })
    return rows


def run_cpu(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    root = project_root()
    command = [
        sys.executable,
        str(root / "bench_resize.py"),
        "--impl",
        args.cpu_impl,
        "--signature",
        args.cpu_signature,
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
    print(f"[run] cpu batch={batch_size:<2d}")
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
        "method": "cpu",
        "batch_size": batch_size,
        "median_ms_per_frame": data["median_ms_per_frame"],
        "median_ms_per_batch": stats["median_ms"],
        "p95_ms_per_batch": stats["p95_ms"],
        "fps_by_median": data["fps_by_median"],
        "input_throughput_GBps": data["input_throughput_GBps"],
    }


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
    csv_path = args.out_dir / "batch_resize_cpu_cuda_baseline.csv"
    json_path = args.out_dir / "batch_resize_cpu_cuda_baseline.json"
    png_path = args.out_dir / "batch_resize_cpu_cuda_baseline.png"

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
                "iters": args.iters,
                "warmup": args.warmup,
                "batch_sizes": args.batch_sizes,
                "input_pool_size": args.input_pool_size,
                "cuda_csv": str(args.cuda_csv),
                "cpu_impl": args.cpu_impl,
            },
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )
    plot_rows(rows, png_path, args.batch_sizes)
    return csv_path, json_path, png_path


def print_summary(rows: list[dict[str, Any]]) -> None:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["batch_size"], {})[row["method"]] = row

    print("\nSummary")
    print("batch, opencv_ms, cpu_ms, cuda_ms, cuda_kernel_ms, cpu_speedup")
    for batch_size in sorted(grouped):
        group = grouped[batch_size]
        if not {"opencv", "cpu", "cuda", "cuda_kernel_only"} <= set(group):
            continue
        opencv = group["opencv"]["median_ms_per_frame"]
        cpu = group["cpu"]["median_ms_per_frame"]
        cuda = group["cuda"]["median_ms_per_frame"]
        kernel = group["cuda_kernel_only"]["median_ms_per_frame"]
        print(
            f"{batch_size}, "
            f"{opencv:.6f}, "
            f"{cpu:.6f}, "
            f"{cuda:.6f}, "
            f"{kernel:.6f}, "
            f"{opencv / cpu:.4f}x"
        )


def main() -> int:
    args = parse_args()
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be positive and --warmup must be non-negative")
    if args.input_pool_size <= 0:
        raise SystemExit("--input-pool-size must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise SystemExit("all batch sizes must be positive")
    if not args.cuda_csv.exists():
        raise SystemExit(f"CUDA CSV not found: {args.cuda_csv}")

    rows = load_cuda_rows(args.cuda_csv, args.batch_sizes)
    for batch_size in args.batch_sizes:
        rows.append(run_cpu(args, batch_size))

    csv_path, json_path, png_path = write_outputs(args, rows)
    print_summary(rows)
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
