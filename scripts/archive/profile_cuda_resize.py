#!/usr/bin/env python3
"""Profile the CUDA resize pipeline stages after warmup."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cuda_resize_py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile cuda_resize_py stage timings.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--src-width", type=int, default=3840)
    parser.add_argument("--src-height", type=int, default=1920)
    parser.add_argument("--dst-width", type=int, default=640)
    parser.add_argument("--dst-height", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def make_input(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    if args.batch_size == 1:
        shape = (args.src_height, args.src_width, 3)
    else:
        shape = (args.batch_size, args.src_height, args.src_width, 3)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[int((len(ordered) - 1) * 0.95)],
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--batch-size and --iters must be positive; --warmup must be non-negative")

    src = make_input(args)
    dsize = (args.dst_width, args.dst_height)

    for _ in range(args.warmup):
        cuda_resize_py.resize(src, dsize, cv2.INTER_LINEAR)

    profiles = []
    checksum = 0
    for _ in range(args.iters):
        output, profile = cuda_resize_py.resize_profile(src, dsize, cv2.INTER_LINEAR)
        checksum = int(output.sum(dtype=np.uint64))
        profiles.append(profile)

    keys = [
        "total_with_numpy_copy_ms_per_frame",
        "pipeline_total_ms_per_frame",
        "input_copy_ms_per_frame",
        "h2d_ms_per_frame",
        "kernel_ms_per_frame",
        "d2h_ms_per_frame",
        "gpu_total_ms_per_frame",
        "output_copy_ms_per_frame",
        "numpy_output_copy_ms_per_frame",
    ]
    result = {
        "config": {
            "batch_size": args.batch_size,
            "src_width": args.src_width,
            "src_height": args.src_height,
            "dst_width": args.dst_width,
            "dst_height": args.dst_height,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
        },
        "checksum": checksum,
        "summary": {
            key: summarize([float(profile[key]) for profile in profiles])
            for key in keys
        },
        "last_profile": profiles[-1],
    }

    print("CUDA resize profile, ms/frame")
    for key in keys:
        stats = result["summary"][key]
        print(
            f"{key}: "
            f"median={stats['median']:.4f}, "
            f"mean={stats['mean']:.4f}, "
            f"p95={stats['p95']:.4f}"
        )
    print(f"checksum: {checksum}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
