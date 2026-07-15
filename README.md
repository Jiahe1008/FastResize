# OpenCV Resize Benchmark

Benchmark case:

- input: fixed random `3840x1920` RGB `uint8`
- output: `640x640` RGB `uint8`
- baseline: `cv2.resize`
- metric: median latency and `fps = 1000 / median_ms`

## Environment

```bash
conda activate hpsc
conda install -c conda-forge opencv numpy
```

## Run OpenCV Baseline

```bash
python bench_resize.py --impl opencv --interp linear --warmup 30 --iters 300
```

Batch baseline uses a Python loop around `cv2.resize` because OpenCV's Python
`resize` API handles one image per call:

```bash
python bench_resize.py --impl opencv --batch-size 8 --interp linear
```

The script prints `median_ms`, `p95_ms`, and `fps_by_median`. Use the median result
for the speedup ratio because it is less sensitive to occasional OS scheduling noise.

## Run a Custom Python-Wrapped Implementation

Wrap any C++/CUDA implementation as a Python function, then benchmark it with the
same script and the same input configuration.

Expected function signatures are selected with `--signature`:

```python
resize(src)
resize(src, dsize)
resize(src, dsize, interpolation)
```

Example:

```bash
python bench_resize.py \
  --impl my_resize_lib:resize \
  --signature src_dsize_interp \
  --interp linear \
  --warmup 30 \
  --iters 300
```

For `--batch-size N` where `N > 1`, the custom function receives one NHWC batch
array:

```text
(N, src_height, src_width, 3)
```

It must return:

```text
(N, dst_height, dst_width, 3)
```

If an implementation only supports single-image input, run it with
`--batch-size 1` or provide a wrapper that loops over the batch.

For non-OpenCV implementations, the script compares output against `cv2.resize` by
default and reports max/mean absolute difference.

## Build CUDA Extension

The included native extensions build Python modules named `cuda_resize_py` and
`fast_cpu_resize`:

```bash
conda activate hpsc
python setup.py build_ext --inplace
```

Run it through the same benchmark:

```bash
python bench_resize.py \
  --impl cuda_resize_py:resize \
  --signature src_dsize_interp \
  --interp linear \
  --warmup 30 \
  --iters 300
```

Batch mode:

```bash
python bench_resize.py \
  --impl cuda_resize_py:resize \
  --signature src_dsize_interp \
  --batch-size 4 \
  --interp linear
```

Run the CPU multithreaded implementation:

```bash
python bench_resize.py \
  --impl fast_cpu_resize:resize \
  --signature src_dsize_interp \
  --interp linear \
  --warmup 30 \
  --iters 300
```

For CUDA implementations, make sure timing includes actual GPU work. Either
synchronize inside your Python-wrapped resize function, or expose a zero-argument
sync function and pass it to the benchmark:

```bash
python bench_resize.py \
  --impl my_cuda_resize:resize \
  --signature src_dsize_interp \
  --sync my_cuda_resize:synchronize
```

## Speedup

Run baseline and custom implementation with identical arguments:

```text
speedup = opencv_median_ms / custom_median_ms
```

For a stable report:

- use a Release/optimized build for the wrapped implementation
- keep image size, interpolation, thread count, seed, and batch size identical
- for CUDA, synchronize before stopping the timer
- run enough iterations, for example `--warmup 30 --iters 300`
- report median call latency, median per-frame latency, p95, FPS, and output difference

## Batch Comparison Plot

Install plotting support if needed:

```bash
conda install -c conda-forge matplotlib
```

Run OpenCV and CUDA across batch sizes `1 2 4 8 16 32 48 64`:

```bash
python scripts/run_batch_resize_compare.py --warmup 30 --iters 300
```

Outputs are written under `results/`:

```text
batch_resize_compare.csv
batch_resize_compare.json
batch_resize_compare.png
```

## CUDA Pipeline Profiling

Break down end-to-end CUDA time into CPU copy, H2D, kernel, D2H, and output copy:

```bash
python scripts/profile_cuda_resize.py --batch-size 1 --warmup 30 --iters 50
```

Useful fields are reported in `ms/frame`, for example:

```text
input_copy_ms_per_frame
h2d_ms_per_frame
kernel_ms_per_frame
d2h_ms_per_frame
output_copy_ms_per_frame
numpy_output_copy_ms_per_frame
```
