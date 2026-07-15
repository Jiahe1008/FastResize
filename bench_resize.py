#!/usr/bin/env python3
"""Benchmark 3840x1920 RGB resize to 640x640 through a Python callable."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import cv2
import numpy as np


ResizeFn = Callable[[Any], Any]


@dataclass(frozen=True)
class BenchConfig:
    src_width: int = 3840
    src_height: int = 1920
    dst_width: int = 640
    dst_height: int = 640
    channels: int = 3
    batch_size: int = 1
    input_pool_size: int = 1
    seed: int = 12345
    warmup: int = 30
    iters: int = 300
    interpolation: str = "linear"
    threads: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fixed random RGB resize input. Baseline is cv2.resize; "
            "custom implementations can be passed as module:function."
        )
    )
    parser.add_argument(
        "--impl",
        default="opencv",
        help="opencv or module:function. The custom function should return dst ndarray.",
    )
    parser.add_argument(
        "--signature",
        choices=("src", "src_dsize", "src_dsize_interp"),
        default="src_dsize_interp",
        help="Call signature for module:function implementations.",
    )
    parser.add_argument(
        "--sync",
        default=None,
        help=(
            "Optional module:function called after each resize before stopping the timer. "
            "Use this for asynchronous CUDA implementations."
        ),
    )
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, default=0, help="cv2 thread count; 0 keeps default.")
    parser.add_argument(
        "--interp",
        choices=("linear", "area", "nearest", "cubic"),
        default="linear",
        help="Interpolation mode. Keep this identical when computing speedup.",
    )
    parser.add_argument("--src-width", type=int, default=3840)
    parser.add_argument("--src-height", type=int, default=1920)
    parser.add_argument("--dst-width", type=int, default=640)
    parser.add_argument("--dst-height", type=int, default=640)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images per resize call. Values >1 pass NHWC batch input.",
    )
    parser.add_argument(
        "--input-pool-size",
        type=int,
        default=1,
        help=(
            "Number of pre-generated inputs to rotate through during timing. "
            "The default 1 preserves the original repeated-input benchmark."
        ),
    )
    parser.add_argument("--check", action="store_true", help="Compare output with cv2.resize.")
    parser.add_argument(
        "--max-diff",
        type=float,
        default=1.0,
        help="Allowed max absolute difference when --check is enabled.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def interpolation_value(name: str) -> int:
    values = {
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
    }
    return values[name]


def input_shape(config: BenchConfig) -> tuple[int, ...]:
    if config.batch_size == 1:
        return (config.src_height, config.src_width, config.channels)
    return (
        config.batch_size,
        config.src_height,
        config.src_width,
        config.channels,
    )


def make_input_pool(config: BenchConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    shape = (config.input_pool_size, *input_shape(config))
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def select_input(input_pool: np.ndarray, index: int) -> np.ndarray:
    return input_pool[index % input_pool.shape[0]]


def load_resize_fn(impl: str, signature: str, config: BenchConfig) -> ResizeFn:
    dsize = (config.dst_width, config.dst_height)
    interp = interpolation_value(config.interpolation)

    if impl == "opencv":
        if config.threads > 0:
            cv2.setNumThreads(config.threads)

        def resize_opencv(src: np.ndarray) -> np.ndarray:
            if src.ndim == 3:
                return cv2.resize(src, dsize, interpolation=interp)
            if src.ndim == 4:
                output = np.empty(
                    (
                        src.shape[0],
                        config.dst_height,
                        config.dst_width,
                        config.channels,
                    ),
                    dtype=src.dtype,
                )
                for index, frame in enumerate(src):
                    output[index] = cv2.resize(frame, dsize, interpolation=interp)
                return output
            raise ValueError(f"expected HWC or NHWC input, got shape {src.shape}")

        return resize_opencv

    if ":" not in impl: 
        raise SystemExit("--impl must be 'opencv' or 'module:function'")

    module_name, func_name = impl.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    if signature == "src":
        return lambda src: func(src)
    if signature == "src_dsize":
        return lambda src: func(src, dsize)
    return lambda src: func(src, dsize, interp)


def load_sync_fn(spec: str | None) -> Callable[[], None] | None:
    if spec is None:
        return None
    if ":" not in spec:
        raise SystemExit("--sync must be module:function")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func


def expected_output_shape(config: BenchConfig) -> tuple[int, ...]:
    image_shape = (config.dst_height, config.dst_width, config.channels)
    if config.batch_size == 1:
        return image_shape
    return (config.batch_size, *image_shape)


def normalize_output(output: Any) -> np.ndarray:
    if isinstance(output, np.ndarray):
        return output
    if isinstance(output, (list, tuple)):
        return np.stack(output, axis=0)
    raise TypeError(f"resize function returned {type(output).__name__}, expected ndarray")


def run_resize(
    resize_fn: ResizeFn,
    sync_fn: Callable[[], None] | None,
    src: np.ndarray,
    impl: str,
    config: BenchConfig,
) -> np.ndarray:
    try:
        output = resize_fn(src)
        if sync_fn is not None:
            sync_fn()
        return normalize_output(output)
    except Exception as exc:
        mode = "batch" if config.batch_size > 1 else "single-image"
        raise SystemExit(
            f"Implementation {impl!r} failed for {mode} input shape {src.shape}. "
            "If this implementation does not support batch input, run with "
            "--batch-size 1 or provide a batch-capable wrapper. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def validate_output(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        hint = ""
        if len(expected.shape) == 4 and len(actual.shape) == 3:
            hint = (
                " The implementation returned a single image for batch input; "
                "run with --batch-size 1 or provide a batch-capable wrapper."
            )
        raise SystemExit(
            f"Output shape mismatch: got {actual.shape}, expected {expected.shape}.{hint}"
        )
    if actual.dtype != expected.dtype:
        raise SystemExit(f"Output dtype mismatch: got {actual.dtype}, expected {expected.dtype}")


def summarize(times_ms: list[float]) -> dict[str, float]:
    ordered = sorted(times_ms)
    n = len(ordered)

    def percentile(p: float) -> float:
        if n == 1:
            return ordered[0]
        pos = (n - 1) * p
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    return {
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


def check_output(
    src: np.ndarray,
    resize_fn: ResizeFn,
    sync_fn: Callable[[], None] | None,
    impl: str,
    config: BenchConfig,
    max_diff: float,
) -> dict[str, float]:
    expected = load_resize_fn("opencv", "src_dsize_interp", config)(src)
    actual = run_resize(resize_fn, sync_fn, src, impl, config)
    validate_output(actual, expected)

    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    result = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }
    if result["max_abs_diff"] > max_diff:
        raise SystemExit(
            f"Output check failed: max_abs_diff={result['max_abs_diff']} > {max_diff}"
        )
    return result


def main() -> int:
    args = parse_args()
    config = BenchConfig(
        src_width=args.src_width,
        src_height=args.src_height,
        dst_width=args.dst_width,
        dst_height=args.dst_height,
        batch_size=args.batch_size,
        input_pool_size=args.input_pool_size,
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
        interpolation=args.interp,
        threads=args.threads,
    )
    if (
        config.iters <= 0 or
        config.warmup < 0 or
        config.batch_size <= 0 or
        config.input_pool_size <= 0
    ):
        raise SystemExit(
            "--iters, --batch-size, and --input-pool-size must be positive; "
            "--warmup must be non-negative"
        )

    input_pool = make_input_pool(config)
    resize_fn = load_resize_fn(args.impl, args.signature, config)
    sync_fn = load_sync_fn(args.sync)

    check = None
    if args.check or args.impl != "opencv":
        check = check_output(
            select_input(input_pool, 0),
            resize_fn,
            sync_fn,
            args.impl,
            config,
            args.max_diff,
        )

    for index in range(config.warmup):
        src = select_input(input_pool, index)
        dst = run_resize(resize_fn, sync_fn, src, args.impl, config)

    times_ms: list[float] = []
    checksum = 0
    for index in range(config.iters):
        src = select_input(input_pool, index)
        start = time.perf_counter_ns()
        dst = run_resize(resize_fn, sync_fn, src, args.impl, config)
        end = time.perf_counter_ns()
        times_ms.append((end - start) / 1_000_000.0)
        checksum = int(dst.sum(dtype=np.uint64))

    stats = summarize(times_ms)
    calls_per_second = 1000.0 / stats["median_ms"]
    fps = config.batch_size * calls_per_second
    median_ms_per_frame = stats["median_ms"] / config.batch_size
    input_gbps = (
        config.src_width * config.src_height * config.channels * fps / 1_000_000_000.0
    )
    result = {
        "impl": args.impl,
        "signature": args.signature if args.impl != "opencv" else "opencv",
        "sync": args.sync,
        "input_shape": input_shape(config),
        "input_pool_shape": tuple(input_pool.shape),
        "expected_output_shape": expected_output_shape(config),
        "config": asdict(config),
        "stats": stats,
        "calls_per_second_by_median": calls_per_second,
        "median_ms_per_frame": median_ms_per_frame,
        "fps_by_median": fps,
        "input_throughput_GBps": input_gbps,
        "checksum": checksum,
        "check": check,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"impl: {result['impl']}")
        print(
            f"case: {config.src_width}x{config.src_height} RGB8 -> "
            f"{config.dst_width}x{config.dst_height}, interp={config.interpolation}"
        )
        print(
            f"batch_size: {config.batch_size}, "
            f"input_pool_size: {config.input_pool_size}, "
            f"input_shape: {input_shape(config)}"
        )
        print(f"iters: {config.iters}, warmup: {config.warmup}, seed: {config.seed}")
        if check is not None:
            print(
                "check: "
                f"max_abs_diff={check['max_abs_diff']:.4f}, "
                f"mean_abs_diff={check['mean_abs_diff']:.4f}"
            )
        print(f"min_ms: {stats['min_ms']:.4f}")
        print(f"median_ms: {stats['median_ms']:.4f}")
        print(f"median_ms_per_frame: {median_ms_per_frame:.4f}")
        print(f"mean_ms: {stats['mean_ms']:.4f}")
        print(f"p95_ms: {stats['p95_ms']:.4f}")
        print(f"calls_per_second_by_median: {calls_per_second:.2f}")
        print(f"fps_by_median: {fps:.2f}")
        print(f"input_throughput_GBps: {input_gbps:.4f}")
        print(f"checksum: {checksum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
