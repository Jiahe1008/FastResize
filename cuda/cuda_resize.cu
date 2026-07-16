/*
cuda kernel resize

input format:
    CV_8UC3, BGR, HWC
    CV_8U 表示每个通道式unsigned char，占1B
    C3    每个像素3个通道
    BGR   三通道分别是blue, green, red
    HWC   内存排列 height, width, channel
*/

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        const cudaError_t error__ = (call);                                \
        if (error__ != cudaSuccess) {                                      \
            throw std::runtime_error(                                      \
                std::string("CUDA error: ") +                              \
                cudaGetErrorString(error__) +                              \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));       \
        }                                                                  \
    } while (0)

namespace {

__device__ __forceinline__
int clampInt(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__
unsigned char floatToU8(float value) {
    int rounded = __float2int_rn(value); // round to nearest
    rounded = rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded);
    return static_cast<unsigned char>(rounded);
    // unsigned char 可表示8-bit 像素值0-256
}

/*
 * 每个 CUDA 线程负责计算一个输出像素。
 *
 * 输入、输出均为：
 *   uint8
 *   3 通道
 *   BGR
 *   HWC 排列
 *
 * src_pitch 和 dst_pitch 是每行实际占用的字节数，
 * 可能大于 width * 3。
 */

 // restrict 修饰符： 是一个给编译器看的优化承诺：
 // 这个指针指向的内存，不会被其它指针别名访问。
 //  srcPitch 表示输入图像一行占多少字节
__global__ void resizeBilinearBgr8Kernel(
    const unsigned char* __restrict__ src,
    std::size_t srcPitch,
    int srcWidth,
    int srcHeight,
    unsigned char* __restrict__ dst,
    std::size_t dstPitch,
    int dstWidth,
    int dstHeight
) {

    /*
        blockIdx是CUDA kernel内置变量。
        表示当前线程所在的 block，在整个 grid 里的编号。
        在kernel内部，每个线程执行相同代码，但内置变量不同。
    */
    const int dstX =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int dstY =
        static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (dstX >= dstWidth || dstY >= dstHeight) {
        return;
    }

    const float scaleX =
        static_cast<float>(srcWidth) /
        static_cast<float>(dstWidth);

    const float scaleY =
        static_cast<float>(srcHeight) /
        static_cast<float>(dstHeight);

    /*
     * half-pixel 映射：
     *
     * srcX = (dstX + 0.5) * scaleX - 0.5
     * srcY = (dstY + 0.5) * scaleY - 0.5
     */
    const float srcX =
        (static_cast<float>(dstX) + 0.5f) * scaleX - 0.5f;
    const float srcY =
        (static_cast<float>(dstY) + 0.5f) * scaleY - 0.5f;

    const int rawX0 = static_cast<int>(floorf(srcX));
    const int rawY0 = static_cast<int>(floorf(srcY));

    const int rawX1 = rawX0 + 1;
    const int rawY1 = rawY0 + 1;

    const float weightX = srcX - static_cast<float>(rawX0);
    const float weightY = srcY - static_cast<float>(rawY0);

    // 边界采用 replicate
    const int x0 = clampInt(rawX0, 0, srcWidth - 1);
    const int x1 = clampInt(rawX1, 0, srcWidth - 1);
    const int y0 = clampInt(rawY0, 0, srcHeight - 1);
    const int y1 = clampInt(rawY1, 0, srcHeight - 1);
    // 越界的情况：放大（scale<1)

    const unsigned char* row0 = src + y0 * srcPitch;
    const unsigned char* row1 = src + y1 * srcPitch;

    unsigned char* outputRow = dst + dstY * dstPitch;

    const int srcOffset0 = x0 * 3;
    const int srcOffset1 = x1 * 3;
    const int dstOffset = dstX * 3;

    // pragma unroll 告知编译器进行循环展开
    #pragma unroll  
    for (int channel = 0; channel < 3; ++channel) {
        const float pixel00 = static_cast<float>(row0[srcOffset0 + channel]);
        const float pixel01 = static_cast<float>(row0[srcOffset1 + channel]);
        const float pixel10 = static_cast<float>(row1[srcOffset0 + channel]);
        const float pixel11 = static_cast<float>(row1[srcOffset1 + channel]);
    
        // 双线性插值
        // fmaf(a,b,c) = a*b + c
        const float top = fmaf(weightX, pixel01 - pixel00, pixel00);
        const float bottom =fmaf(weightX, pixel11 - pixel10, pixel10);
        const float value = fmaf(weightY, bottom - top, top);

        outputRow[dstOffset + channel] = floatToU8(value);
    }
}

}  // namespace

class CudaResizePipeline {
public:
    static constexpr int kSlotCount = 2;

    struct Profile {
        double totalMs = 0.0;
        double inputCopyMs = 0.0;
        double outputCopyMs = 0.0;
        double h2dMs = 0.0;
        double kernelMs = 0.0;
        double d2hMs = 0.0;
        double gpuTotalMs = 0.0;
    };

    CudaResizePipeline(
        int srcWidth,
        int srcHeight,
        int dstWidth,
        int dstHeight
    )
        : srcWidth_(srcWidth),
          srcHeight_(srcHeight),
          dstWidth_(dstWidth),
          dstHeight_(dstHeight),
          srcRowBytes_(
              static_cast<std::size_t>(srcWidth) * 3),
          dstRowBytes_(
              static_cast<std::size_t>(dstWidth) * 3),
          srcBytes_(srcRowBytes_ * srcHeight),
          dstBytes_(dstRowBytes_ * dstHeight) {
        if (srcWidth <= 0 || srcHeight <= 0 ||
            dstWidth <= 0 || dstHeight <= 0) {
            throw std::invalid_argument(
                "Image dimensions must be positive");
        }

        try {
            for (int index = 0; index < kSlotCount; ++index) {
                allocateSlot(slots_[index]);
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~CudaResizePipeline() {
        release();
    }

    // 禁止拷贝构造函数和拷贝复制
    CudaResizePipeline(const CudaResizePipeline&) = delete;
    CudaResizePipeline& operator=(
        const CudaResizePipeline&) = delete;

    /*
     * 批量处理图像。
     *
     * 输入可以是普通 cv::Mat；函数先将每帧复制到 pinned buffer，
     * 然后异步执行：
     *
     * H2D -> resize kernel -> D2H
     *
     * 两个 slot 交替工作。
     */
    void resizeBatch(
        const std::vector<cv::Mat>& inputs,
        std::vector<cv::Mat>& outputs
    ) {
        resizeBatchInternal(inputs, outputs, nullptr, false);
    }

    void resizeBatchRegisteredInput(
        const std::vector<cv::Mat>& inputs,
        std::vector<cv::Mat>& outputs
    ) {
        resizeBatchInternal(inputs, outputs, nullptr, true);
    }

    void resizeBatchProfile(
        const std::vector<cv::Mat>& inputs,
        std::vector<cv::Mat>& outputs,
        Profile& profile
    ) {
        profile = {};
        resizeBatchInternal(inputs, outputs, &profile, false);
    }

    void resizeBatchRegisteredInputProfile(
        const std::vector<cv::Mat>& inputs,
        std::vector<cv::Mat>& outputs,
        Profile& profile
    ) {
        profile = {};
        resizeBatchInternal(inputs, outputs, &profile, true);
    }

private:
    void resizeBatchInternal(
        const std::vector<cv::Mat>& inputs,
        std::vector<cv::Mat>& outputs,
        Profile* profile,
        bool registeredInput
    ) {
        const auto totalBegin = std::chrono::steady_clock::now();
        validateInputs(inputs);

        const std::size_t frameCount = inputs.size();

        outputs.resize(frameCount);

        for (cv::Mat& output : outputs) {
            output.create(  // 分配内存
                dstHeight_,
                dstWidth_,
                CV_8UC3
            );
        }

        /*
         * 前 kSlotCount 次负责填充流水线，
         * 最后 kSlotCount 次负责排空流水线。
         */
        for (std::size_t iteration = 0;
             iteration < frameCount + kSlotCount;
             ++iteration) {

            /*
             * 获取较早提交的结果，同时保证对应 slot
             * 可以被下一帧安全复用。
             */
            if (iteration >= kSlotCount) {
                const std::size_t finishedFrame =
                    iteration - kSlotCount;

                if (finishedFrame < frameCount) {
                    const int slotIndex =
                        static_cast<int>(finishedFrame % kSlotCount);

                    waitAndCopyOutput(
                        slots_[slotIndex],
                        outputs[finishedFrame],
                        profile
                    );
                }
            }

            /*
             * 提交当前帧。
             */
            if (iteration < frameCount) {
                const int slotIndex =static_cast<int>(iteration % kSlotCount);

                Slot& slot = slots_[slotIndex];

                if (registeredInput) {
                    enqueue(
                        slot,
                        inputs[iteration].ptr<unsigned char>(0),
                        inputs[iteration].step,
                        profile
                    );
                } else {
                    copyInputToPinned(
                        inputs[iteration],
                        slot.hostSrc,
                        profile
                    );

                    enqueue(
                        slot,
                        slot.hostSrc,
                        srcRowBytes_,
                        profile
                    );
                }
            }
        }

        const auto totalEnd = std::chrono::steady_clock::now();
        if (profile != nullptr) {
            profile->totalMs =
                std::chrono::duration<double, std::milli>(
                    totalEnd - totalBegin
                ).count();
        }
    }

    struct Slot {
        unsigned char* hostSrc = nullptr;
        unsigned char* hostDst = nullptr;

        unsigned char* deviceSrc = nullptr;
        unsigned char* deviceDst = nullptr;

        std::size_t deviceSrcPitch = 0;
        std::size_t deviceDstPitch = 0;

        cudaStream_t stream = nullptr;
        cudaEvent_t completeEvent = nullptr;

        cudaEvent_t h2dStartEvent = nullptr;
        cudaEvent_t h2dEndEvent = nullptr;
        cudaEvent_t kernelEndEvent = nullptr;
        cudaEvent_t d2hEndEvent = nullptr;
    };

    void allocateSlot(Slot& slot) {
        /*
         * pinned host buffers：
         * cudaMemcpy2DAsync 才有机会真正异步。
         * 分配page-locked memory，适合GPU DMA拷贝
         */
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&slot.hostSrc),
            srcBytes_,
            cudaHostAllocDefault
        ));

        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&slot.hostDst),
            dstBytes_,
            cudaHostAllocDefault
        ));

        /*
         * 使用 pitched device memory。
         * 每行可能存在对齐 padding。
         * 
         * 分配srcHeight_行，每行有效字节srcRowBytes_
         */
        CUDA_CHECK(cudaMallocPitch(
            reinterpret_cast<void**>(&slot.deviceSrc),
            &slot.deviceSrcPitch,
            srcRowBytes_,
            srcHeight_
        ));

        CUDA_CHECK(cudaMallocPitch(
            reinterpret_cast<void**>(&slot.deviceDst),
            &slot.deviceDstPitch,
            dstRowBytes_,
            dstHeight_
        ));

        CUDA_CHECK(cudaStreamCreateWithFlags(
            &slot.stream,
            cudaStreamNonBlocking
        ));

        /*
         * 这个 event 只用于判断流水线槽位是否完成，
         * 不用于计时，因此关闭 timing。
         */
        CUDA_CHECK(cudaEventCreateWithFlags(
            &slot.completeEvent,
            cudaEventDisableTiming
        ));

        CUDA_CHECK(cudaEventCreate(
            &slot.h2dStartEvent
        ));

        CUDA_CHECK(cudaEventCreate(
            &slot.h2dEndEvent
        ));

        CUDA_CHECK(cudaEventCreate(
            &slot.kernelEndEvent
        ));

        CUDA_CHECK(cudaEventCreate(
            &slot.d2hEndEvent
        ));
    }

    void enqueue(
        Slot& slot,
        const unsigned char* hostSrc,
        std::size_t hostPitch,
        Profile* profile
    ) {
        /*
         * 1. pinned host -> pitched device
            异步2维内存拷贝
         */
        if (profile != nullptr) {
            CUDA_CHECK(cudaEventRecord(
                slot.h2dStartEvent,
                slot.stream
            ));
        }

        CUDA_CHECK(cudaMemcpy2DAsync(
            slot.deviceSrc,
            slot.deviceSrcPitch,
            hostSrc,
            hostPitch,
            srcRowBytes_,
            srcHeight_,
            cudaMemcpyHostToDevice,
            slot.stream
        ));

        if (profile != nullptr) {
            CUDA_CHECK(cudaEventRecord(
                slot.h2dEndEvent,
                slot.stream
            ));
        }

        /*
         * 2. CUDA resize
         *
         * block.x 使用 32，使同一个 warp 中的线程
         * 尽量处理连续的输出像素。
         */
        const dim3 block(32, 8);

        const dim3 grid(
            static_cast<unsigned int>(
                (dstWidth_ + block.x - 1) / block.x),
            static_cast<unsigned int>(
                (dstHeight_ + block.y - 1) / block.y)
        );
        // <<<grid, block, sharedMemBytes, stream>>>
        resizeBilinearBgr8Kernel
        <<<grid, block, 0, slot.stream>>>  
        (
            slot.deviceSrc,
            slot.deviceSrcPitch,
            srcWidth_,
            srcHeight_,
            slot.deviceDst,
            slot.deviceDstPitch,
            dstWidth_,
            dstHeight_
        );

        CUDA_CHECK(cudaGetLastError());

        if (profile != nullptr) {
            CUDA_CHECK(cudaEventRecord(
                slot.kernelEndEvent,
                slot.stream
            ));
        }

        /*
         * 3. pitched device -> pinned host
         */
        CUDA_CHECK(cudaMemcpy2DAsync(
            slot.hostDst,
            dstRowBytes_,
            slot.deviceDst,
            slot.deviceDstPitch,
            dstRowBytes_,
            dstHeight_,
            cudaMemcpyDeviceToHost,
            slot.stream
        ));

        if (profile != nullptr) {
            CUDA_CHECK(cudaEventRecord(
                slot.d2hEndEvent,
                slot.stream
            ));
        }

        /*
         * event 会在同一 stream 中前面的工作全部完成后触发。
         */
        CUDA_CHECK(cudaEventRecord(
            slot.completeEvent,
            slot.stream
        ));
    }

    void waitAndCopyOutput(
        Slot& slot,
        cv::Mat& output,
        Profile* profile
    )
    {
        CUDA_CHECK(cudaEventSynchronize(
            slot.completeEvent
        ));

        if (profile != nullptr) {
            float h2dMs = 0.0f;
            float kernelMs = 0.0f;
            float d2hMs = 0.0f;
            float gpuTotalMs = 0.0f;

            CUDA_CHECK(cudaEventElapsedTime(
                &h2dMs,
                slot.h2dStartEvent,
                slot.h2dEndEvent
            ));
            CUDA_CHECK(cudaEventElapsedTime(
                &kernelMs,
                slot.h2dEndEvent,
                slot.kernelEndEvent
            ));
            CUDA_CHECK(cudaEventElapsedTime(
                &d2hMs,
                slot.kernelEndEvent,
                slot.d2hEndEvent
            ));
            CUDA_CHECK(cudaEventElapsedTime(
                &gpuTotalMs,
                slot.h2dStartEvent,
                slot.d2hEndEvent
            ));

            profile->h2dMs += h2dMs;
            profile->kernelMs += kernelMs;
            profile->d2hMs += d2hMs;
            profile->gpuTotalMs += gpuTotalMs;
        }

        const auto copyBegin = std::chrono::steady_clock::now();
        for (int row = 0; row < dstHeight_; ++row) {
            std::memcpy(
                output.ptr<unsigned char>(row),
                slot.hostDst +
                    static_cast<std::size_t>(row) *
                    dstRowBytes_,
                dstRowBytes_
            );
        }
        const auto copyEnd = std::chrono::steady_clock::now();
        if (profile != nullptr) {
            profile->outputCopyMs +=
                std::chrono::duration<double, std::milli>(
                    copyEnd - copyBegin
                ).count();
        }
    }

    void copyInputToPinned(
        const cv::Mat& input,
        unsigned char* pinnedDestination,
        Profile* profile
    )
    const {
        const auto copyBegin = std::chrono::steady_clock::now();
        for (int row = 0; row < srcHeight_; ++row) {
            // memcpy(void* dest, const void* src, size_t n);
            std::memcpy(
                pinnedDestination + static_cast<std::size_t>(row) * srcRowBytes_,
                input.ptr<unsigned char>(row),
                srcRowBytes_
            );
        }
        const auto copyEnd = std::chrono::steady_clock::now();
        if (profile != nullptr) {
            profile->inputCopyMs +=
                std::chrono::duration<double, std::milli>(
                    copyEnd - copyBegin
                ).count();
        }
    }

    void validateInputs(const std::vector<cv::Mat>& inputs) const {
        for (std::size_t index = 0;
             index < inputs.size();
             ++index) {
            const cv::Mat& input = inputs[index];

            if (input.empty()) {
                throw std::invalid_argument(
                    "Input frame " +
                    std::to_string(index) +
                    " is empty");
            }

            if (input.type() != CV_8UC3) {
                throw std::invalid_argument(
                    "Input frame must be CV_8UC3");
            }

            if (input.cols != srcWidth_ ||
                input.rows != srcHeight_) {
                throw std::invalid_argument(
                    "All input frames must have the "
                    "configured source dimensions");
            }
        }
    }

    void release() noexcept {
        /*
         * 析构阶段不能抛异常。
         */
        for (Slot& slot : slots_) {
            if (slot.stream != nullptr) {
                cudaStreamSynchronize(slot.stream);
            }

            if (slot.completeEvent != nullptr) {
                cudaEventDestroy(slot.completeEvent);
                slot.completeEvent = nullptr;
            }

            if (slot.h2dStartEvent != nullptr) {
                cudaEventDestroy(slot.h2dStartEvent);
                slot.h2dStartEvent = nullptr;
            }

            if (slot.h2dEndEvent != nullptr) {
                cudaEventDestroy(slot.h2dEndEvent);
                slot.h2dEndEvent = nullptr;
            }

            if (slot.kernelEndEvent != nullptr) {
                cudaEventDestroy(slot.kernelEndEvent);
                slot.kernelEndEvent = nullptr;
            }

            if (slot.d2hEndEvent != nullptr) {
                cudaEventDestroy(slot.d2hEndEvent);
                slot.d2hEndEvent = nullptr;
            }

            if (slot.stream != nullptr) {
                cudaStreamDestroy(slot.stream);
                slot.stream = nullptr;
            }

            if (slot.deviceSrc != nullptr) {
                cudaFree(slot.deviceSrc);
                slot.deviceSrc = nullptr;
            }

            if (slot.deviceDst != nullptr) {
                cudaFree(slot.deviceDst);
                slot.deviceDst = nullptr;
            }

            if (slot.hostSrc != nullptr) {
                cudaFreeHost(slot.hostSrc);
                slot.hostSrc = nullptr;
            }

            if (slot.hostDst != nullptr) {
                cudaFreeHost(slot.hostDst);
                slot.hostDst = nullptr;
            }
        }
    }

    int srcWidth_;
    int srcHeight_;
    int dstWidth_;
    int dstHeight_;

    std::size_t srcRowBytes_;
    std::size_t dstRowBytes_;
    std::size_t srcBytes_;
    std::size_t dstBytes_;

    Slot slots_[kSlotCount];
};

#ifndef CUDA_RESIZE_NO_MAIN
int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " input.jpg dst_width dst_height "
               "[frame_count]\n";
        return EXIT_FAILURE;
    }

    try {
        const std::string inputPath = argv[1];
        const int dstWidth = std::stoi(argv[2]);
        const int dstHeight = std::stoi(argv[3]);
        const int frameCount =
            argc == 5 ? std::stoi(argv[4]) : 100;

        if (dstWidth <= 0 ||
            dstHeight <= 0 ||
            frameCount <= 0) {
            throw std::invalid_argument(
                "Dimensions and frame count "
                "must be positive");
        }

        cv::Mat input = cv::imread(
            inputPath,
            cv::IMREAD_COLOR
        );

        if (input.empty()) {
            throw std::runtime_error(
                "Failed to read image: " + inputPath);
        }

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        if (deviceCount <= 0) {
            throw std::runtime_error(
                "No CUDA device is available");
        }

        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

        std::cout
            << "GPU: " << properties.name << '\n'
            << "Input: "
            << input.cols << "x" << input.rows << '\n'
            << "Output: "
            << dstWidth << "x" << dstHeight << '\n'
            << "Frames: "
            << frameCount << "\n\n";

        CudaResizePipeline pipeline(
            input.cols,
            input.rows,
            dstWidth,
            dstHeight
        );

        /*
         * vector 中的 cv::Mat 是浅拷贝。
         * 所有输入都只读地引用同一张图，用于吞吐测试。
         */
        std::vector<cv::Mat> frames(
            static_cast<std::size_t>(frameCount),
            input
        );

        std::vector<cv::Mat> outputs;

        /*
         * 预热 CUDA context 和 kernel。
         */
        {
            std::vector<cv::Mat> warmupFrames(4, input);
            std::vector<cv::Mat> warmupOutputs;

            pipeline.resizeBatch(
                warmupFrames,
                warmupOutputs
            );
        }

        const auto begin = std::chrono::steady_clock::now();

        pipeline.resizeBatch(frames, outputs);

        const auto end = std::chrono::steady_clock::now();

        const double totalMilliseconds =
            std::chrono::duration<double, std::milli>(
                end - begin
            ).count();

        const double averageMilliseconds =
            totalMilliseconds /
            static_cast<double>(frameCount);

        const double framesPerSecond =
            1000.0 / averageMilliseconds;

        /*
         * 使用 OpenCV CPU resize 做基本正确性检查。
         * 不要求逐 bit 完全一致，因为插值内部的
         * 浮点计算和舍入方式可能不同。
         */
        cv::Mat cpuReference;

        cv::resize(
            input,
            cpuReference,
            cv::Size(dstWidth, dstHeight),
            0.0,
            0.0,
            cv::INTER_LINEAR
        );

        const double maximumDifference =
            cv::norm(
                cpuReference,
                outputs.front(),
                cv::NORM_INF
            );

        std::cout
            << std::fixed
            << std::setprecision(4)
            << "Total pipeline time: "
            << totalMilliseconds << " ms\n"
            << "Average time:        "
            << averageMilliseconds
            << " ms/frame\n"
            << "Throughput:           "
            << framesPerSecond << " FPS\n"
            << "Max CPU/CUDA diff:    "
            << maximumDifference << '\n';

        if (!cv::imwrite(
                "cuda_resized.jpg",
                outputs.front())) {
            throw std::runtime_error(
                "Failed to save cuda_resized.jpg");
        }

        std::cout
            << "Saved: cuda_resized.jpg\n";

        return EXIT_SUCCESS;
    } catch (const cv::Exception& error) {
        std::cerr
            << "OpenCV error:\n"
            << error.what() << '\n';
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#endif  // CUDA_RESIZE_NO_MAIN
