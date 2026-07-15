# 实验优化过程记录

本文简要记录 OpenCV resize 优化实验的演进过程。目标任务是对 `3840x1920` RGB `uint8` 图像做双线性 resize，默认测试到 `640x640`，并以 OpenCV `cv2.resize` 作为 baseline。

## 1. Python Benchmark 基线

最开始实现了 [bench_resize.py](bench_resize.py)，用于统一测试不同 Python 可调用 resize 方法。

基线方法是：

```text
cv2.resize(src, dsize, interpolation=cv2.INTER_LINEAR)
```

主要指标包括：

```text
median_ms
median_ms_per_frame
p95_ms
fps_by_median
input_throughput_GBps
checksum
```

后来加入了 `--batch-size`，使自定义方法可以接收 NHWC batch 输入：

```text
(N, H, W, 3)
```

OpenCV Python 本身不支持 batch resize，因此 baseline 的 batch 实现是 Python 循环调用 `cv2.resize`。

## 2. CUDA Resize 初版

实现了 CUDA 双线性 resize kernel，文件主要在：

```text
cuda/cuda_resize.cu
cuda/python_cuda_resize.cu
```

CUDA 版本封装成 Python 模块：

```text
cuda_resize_py:resize
```

早期 CUDA 路径包含：

```text
NumPy input
-> CPU pinned buffer copy
-> H2D
-> CUDA resize kernel
-> D2H
-> NumPy output
```

测试发现 CUDA kernel 本身很快，但 end-to-end 速度不理想，主要瓶颈在 CPU/GPU 传输和额外内存拷贝。

## 3. CUDA Profiling 与传输优化

增加了 CUDA profiling，历史脚本保存在：

```text
scripts/archive/profile_cuda_resize.py
```

profiling 拆分了：

```text
input_copy_ms_per_frame
h2d_ms_per_frame
kernel_ms_per_frame
d2h_ms_per_frame
output_copy_ms_per_frame
numpy_output_copy_ms_per_frame
```

重要发现：

```text
CUDA kernel-only 通常约 0.07~0.12 ms/frame
CUDA end-to-end 通常数 ms/frame
```

说明 kernel 不是主要瓶颈，传输和封装成本才是主要瓶颈。

之后使用 `cudaHostRegister` 注册 NumPy 输入，去掉了 NumPy 到 pinned host buffer 的额外拷贝。

## 4. CUDA Pool 输入问题

为了让 benchmark 更接近视频流，给 [bench_resize.py](bench_resize.py) 增加了：

```text
--input-pool-size
```

它会预生成多组输入，并在计时循环中轮换使用，避免一直 resize 同一张热缓存图片。

加入 input pool 后发现 CUDA end-to-end 明显变慢。原因之一是 `cuda_resize_py` 原来只缓存一个 registered input pointer。pool 轮换时会反复：

```text
cudaHostUnregister
cudaHostRegister
```

因此后来把 `RegisteredInput` 从单个缓存改成多个 pointer 缓存，并新增：

```text
cuda_resize_py.clear_registered_inputs()
```

用于显式释放已注册的输入。

## 5. CPU C++ 浮点版本

实现了 CPU C++ resize 模块：

```text
cpu/python_fast_cpu_resize.cpp
```

封装为：

```text
fast_cpu_resize:resize_float
```

主要优化：

```text
预计算 x/y 映射表
batch + 单帧内部并行
释放 Python GIL
多线程处理输出行
```

该版本仍然使用 float 做双线性插值，作为后续定点优化的 reference。

## 6. CPU C++ 定点整数版本

在保留旧 float 实现的基础上，新增定点整数路径：

```text
fast_cpu_resize:resize
fast_cpu_resize:resize_fixed
```

主要变化：

```text
float weight -> fixed-point weight
float 插值 -> int 乘加和右移
减少 float 运算和 lrintf 开销
```

同时加入了一个通用快路径：

```text
如果 wy == 0 或 y0 == y1，只做水平插值
```

这个快路径不是硬编码尺寸，而是根据映射表权重自动触发。对于 `3840x1920 -> 640x640` 这种比例，Y 方向经常不需要真正插值，因此收益明显。

当前观察结果是：

```text
batch=1 仍然不如 OpenCV baseline
batch>1 时 cpu_fixed 明显快于 OpenCV Python batch loop
```

但需要注意：OpenCV Python 不支持 native batch，所以 batch>1 的 speedup 包含了 batch API 和 Python 调用次数减少的收益，不能简单解释为单帧 resize kernel 比 OpenCV 更快。

## 7. Benchmark 公平性调整

加入 `input_pool_size` 后，测试更接近流式输入：

```text
input_pool_size=1: 反复使用同一组输入
input_pool_size>1: 在多组预生成输入之间轮换
```

这样可以减少单张图片热缓存带来的偏差。

但 pool 也会增加内存占用，例如：

```text
batch=64, pool=2
约等于 128 张 4K RGB 输入
```

因此正式测试建议先使用：

```text
--input-pool-size 2
```

## 8. 脚本整理

早期脚本较多，分别对应不同阶段：

```text
CUDA 初版对比
CUDA 优化前后对比
CUDA kernel-only profiling
CPU/CUDA/baseline 混合画图
只重跑 baseline 并替换结果
```

这些历史脚本已经归档到：

```text
scripts/archive/
```

当前主流程只保留两个脚本：

```text
scripts/run_resize_suite.py
scripts/plot_resize_results.py
```

[scripts/run_resize_suite.py](scripts/run_resize_suite.py) 负责批量运行并输出：

```text
CSV
JSON
```

[scripts/plot_resize_results.py](scripts/plot_resize_results.py) 只读取 CSV 并生成 PNG。

当前统一记录五条线：

```text
baseline          OpenCV cv2.resize
cpu_float         CPU C++ float bilinear
cpu_fixed         CPU C++ fixed-point bilinear
cuda              CUDA end-to-end
cuda_resize_only  CUDA kernel-only resize time
```

## 9. 当前结论

目前结果可以概括为：

```text
OpenCV batch=1 很强，当前 CPU/CUDA end-to-end 难以超过
CPU fixed-point 在 batch workload 下有明显优势
CUDA kernel-only 很快，但 end-to-end 主要受 H2D/D2H 和封装开销限制
batch>1 的对比应解释为 Python-callable batch pipeline 对 OpenCV Python loop 的加速
```

更严谨的报告表述应区分：

```text
单帧 resize kernel 对比
batch pipeline end-to-end 对比
CUDA kernel-only 局部计时
```

## 10. 后续方向

CPU 方向：

```text
实现持久化 thread pool，降低 batch=1 的线程创建开销
继续做 SIMD / row buffer / separable bilinear
```

CUDA 方向：

```text
继续优化 H2D/D2H
减少 Python/NumPy 边界开销
如果上游能在 GPU 解码 MJPEG，则 CUDA pipeline 才更容易体现优势
```

