#!/usr/bin/env python3
"""Run the resize benchmark suite and write CSV/JSON results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]
DEFAULT_METHODS = [
    "baseline",
    "cpu_float",
    "cpu_fixed",
    "cpu_simd",
    "cuda",
    "cuda_resize_only",
]
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
        description=(
            "Run baseline, CPU float, CPU fixed-point, CPU SIMD, "
            "CUDA end-to-end, and CUDA resize-only benchmarks."
        )
    )
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--kernel-iters", type=int, default=50)
    parser.add_argument("--kernel-warmup", type=int, default=30)
    parser.add_argument("--src-width", type=int, default=3840)
    parser.add_argument("--src-height", type=int, default=1920)
    parser.add_argument("--dst-width", type=int, default=640)
    parser.add_argument("--dst-height", type=int, default=640)
    parser.add_argument("--interp", default="linear", choices=("linear",))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--input-pool-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=None,
        help=(
            "Optional existing CSV to merge from. Rows for methods selected by "
            "--methods and tested batch sizes are replaced; other rows are kept."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results_cpu"))
    parser.add_argument("--basename", default="resize_suite")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def interpolation_value(name: str) -> int:
    if name != "linear":
        raise ValueError("this suite currently supports only linear interpolation")
    return cv2.INTER_LINEAR


def extract_json(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"benchmark did not print JSON:\n{stdout}")
    return json.loads(stdout[start : end + 1])


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def run_bench_method(args: argparse.Namespace, method: str, batch_size: int) -> dict[str, Any]:
    impl_by_method = {
        "baseline": "opencv",
        "cpu_float": "fast_cpu_resize:resize_float",
        "cpu_fixed": "fast_cpu_resize:resize",
        "cpu_simd": "fast_cpu_resize_simd:resize",
        "cuda": "cuda_resize_py:resize",
    }
    root = project_root()
    command = [
        sys.executable,
        str(root / "bench_resize.py"),
        "--impl",
        impl_by_method[method],
        "--signature",
        "src_dsize_interp",
        "--interp",
        args.interp,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--batch-size",
        str(batch_size),
        "--input-pool-size",
        str(args.input_pool_size),
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
        "--json",
    ]
    print(f"[run] {method:16s} batch={batch_size:<2d}", flush=True)
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


def input_shape(args: argparse.Namespace, batch_size: int) -> tuple[int, ...]:
    if batch_size == 1:
        return (args.src_height, args.src_width, 3)
    return (batch_size, args.src_height, args.src_width, 3)


def make_input_pool(args: argparse.Namespace, batch_size: int) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    shape = (args.input_pool_size, *input_shape(args, batch_size))
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def run_cuda_resize_only(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import cuda_resize_py

    try:
        src_pool = make_input_pool(args, batch_size)
        dsize = (args.dst_width, args.dst_height)
        interpolation = interpolation_value(args.interp)

        for index in range(args.kernel_warmup):
            src = src_pool[index % args.input_pool_size]
            cuda_resize_py.resize(src, dsize, interpolation)

        values: list[float] = []
        print(f"[run] {'cuda_resize_only':16s} batch={batch_size:<2d}", flush=True)
        for index in range(args.kernel_iters):
            src = src_pool[index % args.input_pool_size]
            _, profile = cuda_resize_py.resize_profile(src, dsize, interpolation)
            values.append(float(profile["kernel_ms_per_frame"]))
    finally:
        if hasattr(cuda_resize_py, "clear_registered_inputs"):
            cuda_resize_py.clear_registered_inputs()

    median = statistics.median(values)
    fps = 1000.0 / median
    return {
        "method": "cuda_resize_only",
        "batch_size": batch_size,
        "median_ms_per_frame": median,
        "median_ms_per_batch": median * batch_size,
        "p95_ms_per_batch": percentile(values, 0.95) * batch_size,
        "fps_by_median": fps,
        "input_throughput_GBps": (
            args.src_width * args.src_height * 3 * fps / 1_000_000_000.0
        ),
    }


def run_method(args: argparse.Namespace, method: str, batch_size: int) -> dict[str, Any]:
    if method == "cuda_resize_only":
        return run_cuda_resize_only(args, batch_size)
    return run_bench_method(args, method, batch_size)


def load_base_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as file:
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


def merge_rows(
    base_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    methods: list[str],
    batch_sizes: list[int],
) -> list[dict[str, Any]]:
    replaced = {
        (method, batch_size)
        for method in methods
        for batch_size in batch_sizes
    }
    rows = [
        row for row in base_rows
        if (row["method"], row["batch_size"]) not in replaced
    ]
    rows.extend(new_rows)
    return rows


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.basename}.csv"
    json_path = args.out_dir / f"{args.basename}.json"

    order = {method: index for index, method in enumerate(DEFAULT_METHODS)}
    rows = sorted(rows, key=lambda row: (row["batch_size"], order[row["method"]]))

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps({
            "config": {
                "iters": args.iters,
                "warmup": args.warmup,
                "kernel_iters": args.kernel_iters,
                "kernel_warmup": args.kernel_warmup,
                "src_width": args.src_width,
                "src_height": args.src_height,
                "dst_width": args.dst_width,
                "dst_height": args.dst_height,
                "interp": args.interp,
                "seed": args.seed,
                "input_pool_size": args.input_pool_size,
                "batch_sizes": args.batch_sizes,
                "methods": args.methods,
                "base_csv": None if args.base_csv is None else str(args.base_csv),
            },
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


def print_summary(rows: list[dict[str, Any]]) -> None:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["batch_size"], {})[row["method"]] = row

    print("\nSummary")
    print("batch, baseline_ms, cpu_float_ms, cpu_fixed_ms, cpu_simd_ms, cuda_ms, cuda_resize_only_ms")
    for batch_size in sorted(grouped):
        group = grouped[batch_size]
        values = [
            group.get(method, {}).get("median_ms_per_frame")
            for method in DEFAULT_METHODS
        ]
        formatted = [
            "NA" if value is None else f"{value:.6f}"
            for value in values
        ]
        print(f"{batch_size}, " + ", ".join(formatted))


def main() -> int:
    args = parse_args()
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be positive and --warmup must be non-negative")
    if args.kernel_iters <= 0 or args.kernel_warmup < 0:
        raise SystemExit("--kernel-iters must be positive and --kernel-warmup must be non-negative")
    if args.input_pool_size <= 0:
        raise SystemExit("--input-pool-size must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise SystemExit("all batch sizes must be positive")
    if args.base_csv is not None and not args.base_csv.exists():
        raise SystemExit(f"--base-csv not found: {args.base_csv}")

    new_rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        for method in args.methods:
            new_rows.append(run_method(args, method, batch_size))

    if args.base_csv is None:
        rows = new_rows
    else:
        rows = merge_rows(
            load_base_rows(args.base_csv),
            new_rows,
            args.methods,
            args.batch_sizes,
        )

    csv_path, json_path = write_outputs(args, rows)
    print_summary(rows)
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
