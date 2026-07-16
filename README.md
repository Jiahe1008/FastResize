# OpenCV Resize 性能优化

本项目研究 RGB `uint8` 图像双线性 resize 的通用性能优化。默认测试场景为：

```text
输入：3840 × 1920 × 3，RGB uint8
输出：640 × 640 × 3，RGB uint8
插值：双线性插值
基线：OpenCV cv2.resize(..., interpolation=cv2.INTER_LINEAR)
调用入口：Python
```

自定义实现使用 C++ 或 CUDA 编写，并统一封装成 Python 函数进行测试。优化不固定输入和输出尺寸，但只针对三通道 `uint8` 双线性 resize。

## 当前结论

当前推荐方案是 CPU C++ 定点批处理实现：

- `batch=1` 时，OpenCV 单帧 resize 仍然最快。
- `batch>=2` 时，CPU 定点批处理相对 OpenCV Python 逐帧循环获得约 `2.06x ~ 2.90x` 加速。
- CUDA resize 内核本身明显快于 OpenCV，但 CPU 到 GPU、GPU 到 CPU 的传输使端到端方案慢于基线。
- 独立 AVX2 SIMD 版本没有超过 CPU 定点版本，作为探索性负结果保留。

需要注意：OpenCV Python 的 `resize` 不支持原生 batch。`batch>1` 的 baseline 是 Python 循环调用多次 `cv2.resize`，因此 batch 加速比表示完整 Python 可调用批处理流水线的收益，不能等同于自定义单帧内核全面超过 OpenCV。

## 环境

进入项目使用的 Conda 环境：

```bash
conda activate hpsc
```

主要依赖：

- Python 3.11
- NumPy
- OpenCV Python 与 C++ 开发库
- Matplotlib
- 支持 C++17 的编译器
- CUDA Toolkit 和 `nvcc`（构建 CUDA 模块时需要）

如需安装 Python 依赖：

```bash
conda install -c conda-forge numpy opencv matplotlib setuptools
```

## 构建扩展

当前构建会生成三个 Python 扩展模块：

| 模块 | 作用 |
| --- | --- |
| `fast_cpu_resize` | CPU 浮点与 CPU 定点实现 |
| `fast_cpu_resize_simd` | 独立 AVX2 SIMD 探索实现 |
| `cuda_resize_py` | CUDA 端到端 resize 与 profiling |

构建命令：

```bash
conda activate hpsc
python setup.py build_ext --inplace
```

构建成功后，扩展模块会出现在项目根目录，可直接被 Python import。

## 方法说明

### OpenCV baseline

单帧直接调用 `cv2.resize`。batch 输入通过 Python 循环逐帧调用 OpenCV，用作实际 Python 接口基线。

### CPU float

Python 方法：

```text
fast_cpu_resize:resize_float
```

主要用于保留初始 C++ 浮点实现作为参考，包含：

- X/Y 坐标映射预计算
- batch 与单帧输出行并行
- C++ 内部统一调度
- 执行计算时释放 Python GIL

### CPU fixed

Python 方法：

```text
fast_cpu_resize:resize
fast_cpu_resize:resize_fixed
```

这是当前推荐实现，主要优化包括：

- 将浮点权重转换为定点整数权重
- 使用整数乘加、舍入和位移完成插值
- 使用持久化线程池，避免每次调用创建和销毁线程
- 将双线性插值拆分为水平插值和垂直混合
- batch 维度和单帧输出行联合并行
- 当垂直权重为零或上下源行相同时跳过垂直混合

默认测试尺寸与 OpenCV 输出的 `max_abs_diff=0`；非整数缩放比例测试通常满足 `max_abs_diff<=1`。

### CPU SIMD

Python 方法：

```text
fast_cpu_resize_simd:resize
```

该方法尝试使用 AVX2 优化水平定点乘加、垂直混合和结果打包。正式测试中整体慢于 `cpu_fixed`，主要原因是 RGB 交错存储导致水平插值需要按偏移收集输入，向量构造和重排成本抵消了 SIMD 算术收益。因此它不是当前推荐结果。

### CUDA

Python 方法：

```text
cuda_resize_py:resize
cuda_resize_py:resize_profile
```

CUDA 实现包含：

- 每个 CUDA thread 计算一个输出像素
- half-pixel 坐标映射与双线性插值
- pitched device memory
- CUDA stream 与 event
- 异步 H2D、kernel、D2H 流水
- 使用 `cudaHostRegister` 直接注册 NumPy 输入
- 缓存多个已注册输入指针，支持 input pool 轮换

`resize_profile` 可分别观察 H2D、kernel、D2H 等阶段。当前实验中 CUDA kernel-only 约为 `0.07 ~ 0.12 ms/frame`，但端到端约为 `1.92 ~ 2.08 ms/frame`，瓶颈主要是 CPU/GPU 数据传输。

## 单方法 Benchmark

统一入口是 `bench_resize.py`。随机输入在计时前生成，正式计时只覆盖被测 resize 调用和必要的 CUDA 同步。

### OpenCV baseline

```bash
python bench_resize.py \
  --impl opencv \
  --interp linear \
  --batch-size 1 \
  --input-pool-size 2 \
  --warmup 30 \
  --iters 300
```

### CPU float

```bash
python bench_resize.py \
  --impl fast_cpu_resize:resize_float \
  --signature src_dsize_interp \
  --interp linear \
  --batch-size 8 \
  --input-pool-size 2 \
  --warmup 30 \
  --iters 300 \
  --check
```

### CPU fixed

```bash
python bench_resize.py \
  --impl fast_cpu_resize:resize \
  --signature src_dsize_interp \
  --interp linear \
  --batch-size 8 \
  --input-pool-size 2 \
  --warmup 30 \
  --iters 300 \
  --check
```

### CUDA 端到端

```bash
python bench_resize.py \
  --impl cuda_resize_py:resize \
  --signature src_dsize_interp \
  --interp linear \
  --batch-size 8 \
  --input-pool-size 2 \
  --warmup 30 \
  --iters 300 \
  --check
```

当前 CUDA `resize` 返回 CPU NumPy 输出，因此这条命令统计的是包含 H2D、kernel、D2H 在内的端到端耗时。

### 自定义尺寸

```bash
python bench_resize.py \
  --impl fast_cpu_resize:resize \
  --signature src_dsize_interp \
  --src-width 1920 \
  --src-height 1080 \
  --dst-width 1280 \
  --dst-height 720 \
  --batch-size 4 \
  --check
```

## 输入和输出约定

单帧输入：

```text
shape: (H, W, 3)
dtype: uint8
layout: C-contiguous
```

当 `batch_size>1` 时：

```text
输入 shape: (N, H, W, 3)
输出 shape: (N, dst_height, dst_width, 3)
```

`batch_size=1` 使用 HWC 单帧输入，而不是 `(1, H, W, 3)`。

RGB 和 BGR 都是三通道交错数组，resize 对各通道独立计算，因此通道顺序不影响 resize 算法和性能；调用方必须保证输入与输出采用同一种通道语义。

## 关键参数

| 参数 | 含义 |
| --- | --- |
| `--batch-size` | 每次函数调用处理的帧数 |
| `--input-pool-size` | 预生成并轮换使用的输入组数 |
| `--warmup` | 正式计时前的预热调用次数 |
| `--iters` | 正式计时的函数调用次数 |
| `--threads` | OpenCV baseline 使用的线程数；`0` 保持默认值 |
| `--check` | 与 OpenCV 输出比较 |
| `--max-diff` | 正确性检查允许的最大绝对误差 |
| `--json` | 将单次 benchmark 结果打印为 JSON |

实际处理帧数为：

```text
total_frames = iters × batch_size
```

例如 `--iters 300 --batch-size 4` 表示正式计时阶段共处理 `1200` 帧。

`input_pool_size=2` 表示预先生成两组输入，在预热和计时过程中交替使用。它可以减少反复访问同一地址带来的热缓存偏差，但也会增加内存占用。`--basename` 只决定输出文件名，与 input pool 大小无关。

## 输出指标

| 指标 | 含义 |
| --- | --- |
| `median_ms` | 一次函数调用，即一个 batch 的中位耗时 |
| `median_ms_per_frame` | `median_ms / batch_size`，主要比较指标 |
| `mean_ms` | 所有调用耗时的算术平均值 |
| `p95_ms` | 95% 的调用不超过该耗时，用于观察尾延迟 |
| `calls_per_second_by_median` | 按中位耗时计算的每秒函数调用数 |
| `fps_by_median` | 按每帧中位耗时计算的聚合帧率 |
| `input_throughput_GBps` | 按输入图像字节数和 fps 推导的输入吞吐 |
| `max_abs_diff` | 相对 OpenCV 输出的最大绝对差 |
| `mean_abs_diff` | 相对 OpenCV 输出的平均绝对差 |
| `checksum` | 对最后一次输出求和，用于确认结果被实际读取 |

加速比统一使用每帧中位延迟计算：

```text
speedup = baseline_median_ms_per_frame / optimized_median_ms_per_frame
```

`speedup>1` 表示优化方法更快，`speedup<1` 表示优化方法更慢。

## 批量运行与结果汇总

当前主结果只比较 `baseline`、`cpu_float` 和 `cpu_fixed`：

```bash
python scripts/run_resize_suite.py \
  --methods baseline cpu_float cpu_fixed \
  --batch-sizes 1 2 4 8 16 32 48 64 \
  --input-pool-size 2 \
  --warmup 30 \
  --iters 300 \
  --out-dir results_cpu \
  --basename resize_suite_pool3
```

批量脚本会生成：

```text
results_cpu/resize_suite_pool3.csv
results_cpu/resize_suite_pool3.json
```

绘图命令：

```bash
python scripts/plot_resize_results.py \
  --csv results_cpu/resize_suite_pool3.csv \
  --png results_cpu/resize_suite_pool3.png \
  --title "CPU Resize Performance vs Batch Size"
```

当前整理后的 `resize_suite_pool3.csv` 和图片只包含三种方法。现有 CSV 沿用 `input_pool_size=2` 的正式实验数据；文件名中的 `pool3` 是结果 basename，不表示输入池大小为 3。

### 完整研究方法

如需重新测试 CUDA 和 SIMD 探索方法：

```bash
python scripts/run_resize_suite.py \
  --methods baseline cpu_float cpu_fixed cpu_simd cuda cuda_resize_only \
  --batch-sizes 1 2 4 8 16 32 48 64 \
  --input-pool-size 2 \
  --warmup 30 \
  --iters 300 \
  --kernel-warmup 30 \
  --kernel-iters 300 \
  --out-dir results_cpu \
  --basename resize_suite_full
```

### 只重跑部分方法

`--base-csv` 可以复用已有结果，只替换指定方法和 batch size：

```bash
python scripts/run_resize_suite.py \
  --base-csv results_cpu/resize_suite_pool3.csv \
  --methods cpu_fixed \
  --batch-sizes 1 2 4 8 16 32 48 64 \
  --input-pool-size 2 \
  --warmup 30 \
  --iters 300 \
  --out-dir results_cpu \
  --basename resize_suite_pool3
```

未重跑的方法会从 base CSV 保留，`method + batch_size` 相同的旧行会被新结果替换。

## 当前 CPU 结果

数据来自当前 `resize_suite_pool3.csv`，单位为 `ms/frame`：

| Batch | OpenCV baseline | CPU float | CPU fixed | CPU fixed speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.349 | 1.148 | 0.595 | 0.59x |
| 2 | 0.836 | 0.820 | 0.406 | 2.06x |
| 4 | 0.765 | 0.662 | 0.309 | 2.48x |
| 8 | 0.767 | 0.666 | 0.308 | 2.49x |
| 16 | 0.714 | 0.625 | 0.286 | 2.50x |
| 32 | 0.787 | 0.634 | 0.349 | 2.26x |
| 48 | 0.804 | 0.547 | 0.306 | 2.63x |
| 64 | 0.874 | 0.532 | 0.302 | 2.90x |

结果图：

![CPU resize batch benchmark](results_cpu/resize_suite_pool3.png)

对结果的正确解释是：

- OpenCV 在 `batch=1` 时具有最优单帧延迟。
- CPU fixed 在 `batch>=2` 时具有更高的 Python 端到端批处理吞吐。
- batch 加速包含减少 Python 调用次数和 C++ 内部统一调度的收益。
- 多帧实时视频和多路视频预处理是当前 CPU fixed 最有价值的应用场景。

## 项目结构

```text
.
├── bench_resize.py                 # 单方法统一 benchmark
├── setup.py                        # C++/CUDA Python 扩展构建
├── cpu/
│   ├── python_fast_cpu_resize.cpp  # CPU float 与 CPU fixed
│   └── python_fast_cpu_resize_simd.cpp
├── cuda/
│   ├── cuda_resize.cu              # CUDA kernel 与 pipeline
│   └── python_cuda_resize.cu       # Python 扩展接口
├── scripts/
│   ├── run_resize_suite.py         # 批量运行并生成 CSV/JSON
│   ├── plot_resize_results.py      # 读取 CSV 生成 PNG
│   └── archive/                    # 历史实验脚本
├── results_cpu/                    # CPU 主结果
└── history.md                      # 实验与优化过程记录
```

## 实验记录

完整优化过程、性能瓶颈和负结果记录在 [history.md](history.md)。后续每次算法或实现优化都应同步补充该文件。
