#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

constexpr int kWeightBits = 11;
constexpr int kWeightScale = 1 << kWeightBits;
constexpr int kHorizontalRounding = 1 << (kWeightBits - 1);
constexpr int kBilinearRounding = 1 << (2 * kWeightBits - 1);

struct AxisTable {
    std::vector<int> lower;
    std::vector<int> upper;
    std::vector<int> weightFixed;
    std::vector<int> lowerOffset;
    std::vector<int> upperOffset;
    std::vector<int> weightFixedByValue;
    std::vector<int> invWeightFixedByValue;
};

struct ResizeConfig {
    int batchSize = 1;
    int srcWidth = 0;
    int srcHeight = 0;
    int dstWidth = 0;
    int dstHeight = 0;
    bool batched = false;
};

using ResizeWorker = void (*)(
    const std::uint8_t*,
    std::uint8_t*,
    const ResizeConfig&,
    const AxisTable&,
    const AxisTable&,
    int,
    int
);

AxisTable makeAxisTable(int srcSize, int dstSize) {
    AxisTable table;
    table.lower.resize(static_cast<std::size_t>(dstSize));
    table.upper.resize(static_cast<std::size_t>(dstSize));
    table.weightFixed.resize(static_cast<std::size_t>(dstSize));
    table.lowerOffset.resize(static_cast<std::size_t>(dstSize) * 3);
    table.upperOffset.resize(static_cast<std::size_t>(dstSize) * 3);
    table.weightFixedByValue.resize(static_cast<std::size_t>(dstSize) * 3);
    table.invWeightFixedByValue.resize(static_cast<std::size_t>(dstSize) * 3);

    const float scale =
        static_cast<float>(srcSize) / static_cast<float>(dstSize);

    for (int dst = 0; dst < dstSize; ++dst) {
        const float src = (static_cast<float>(dst) + 0.5f) * scale - 0.5f;
        const int rawLower = static_cast<int>(std::floor(src));
        const int rawUpper = rawLower + 1;
        const float weight = src - static_cast<float>(rawLower);

        table.lower[static_cast<std::size_t>(dst)] =
            std::clamp(rawLower, 0, srcSize - 1);
        table.upper[static_cast<std::size_t>(dst)] =
            std::clamp(rawUpper, 0, srcSize - 1);
        const int weightFixed = std::clamp(
            static_cast<int>(std::lrintf(weight * kWeightScale)),
            0,
            kWeightScale
        );
        table.weightFixed[static_cast<std::size_t>(dst)] = weightFixed;

        for (int channel = 0; channel < 3; ++channel) {
            const auto valueIndex = static_cast<std::size_t>(dst * 3 + channel);
            table.lowerOffset[valueIndex] =
                table.lower[static_cast<std::size_t>(dst)] * 3 + channel;
            table.upperOffset[valueIndex] =
                table.upper[static_cast<std::size_t>(dst)] * 3 + channel;
            table.weightFixedByValue[valueIndex] = weightFixed;
            table.invWeightFixedByValue[valueIndex] =
                kWeightScale - weightFixed;
        }
    }

    return table;
}

inline std::uint8_t intToU8(int value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

void horizontalFixedRow(
    const std::uint8_t* srcRow,
    int* scaledRow,
    const ResizeConfig& config,
    const AxisTable& xTable
) {
    const int count = config.dstWidth * 3;
    int index = 0;

#if defined(__AVX2__)
    for (; index + 8 <= count; index += 8) {
        const int* lower = xTable.lowerOffset.data() + index;
        const int* upper = xTable.upperOffset.data() + index;

        const __m256i p0 = _mm256_setr_epi32(
            srcRow[lower[0]],
            srcRow[lower[1]],
            srcRow[lower[2]],
            srcRow[lower[3]],
            srcRow[lower[4]],
            srcRow[lower[5]],
            srcRow[lower[6]],
            srcRow[lower[7]]
        );
        const __m256i p1 = _mm256_setr_epi32(
            srcRow[upper[0]],
            srcRow[upper[1]],
            srcRow[upper[2]],
            srcRow[upper[3]],
            srcRow[upper[4]],
            srcRow[upper[5]],
            srcRow[upper[6]],
            srcRow[upper[7]]
        );
        const __m256i wx = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                xTable.weightFixedByValue.data() + index
            )
        );
        const __m256i invWx = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                xTable.invWeightFixedByValue.data() + index
            )
        );
        const __m256i value = _mm256_add_epi32(
            _mm256_mullo_epi32(p0, invWx),
            _mm256_mullo_epi32(p1, wx)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(scaledRow + index),
            value
        );
    }
#endif

    for (; index < count; ++index) {
        scaledRow[index] =
            srcRow[xTable.lowerOffset[static_cast<std::size_t>(index)]] *
                xTable.invWeightFixedByValue[static_cast<std::size_t>(index)] +
            srcRow[xTable.upperOffset[static_cast<std::size_t>(index)]] *
                xTable.weightFixedByValue[static_cast<std::size_t>(index)];
    }
}

void storeHorizontalScaledU8(const int* scaledRow, std::uint8_t* outRow, int count) {
    int index = 0;

#if defined(__AVX2__)
    const __m256i rounding = _mm256_set1_epi32(kHorizontalRounding);
    const __m128i zero = _mm_setzero_si128();
    for (; index + 8 <= count; index += 8) {
        __m256i value = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(scaledRow + index)
        );
        value = _mm256_add_epi32(value, rounding);
        value = _mm256_srai_epi32(value, kWeightBits);

        const __m128i lo = _mm256_castsi256_si128(value);
        const __m128i hi = _mm256_extracti128_si256(value, 1);
        const __m128i packed16 = _mm_packus_epi32(lo, hi);
        const __m128i packed8 = _mm_packus_epi16(packed16, zero);
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(outRow + index),
            packed8
        );
    }
#endif

    for (; index < count; ++index) {
        outRow[index] = intToU8(
            (scaledRow[index] + kHorizontalRounding) >>
            kWeightBits
        );
    }
}

void storeVerticalMixedU8(
    const int* row0,
    const int* row1,
    std::uint8_t* outRow,
    int count,
    int wy,
    int invWy
) {
    int index = 0;

#if defined(__AVX2__)
    const __m256i wyVector = _mm256_set1_epi32(wy);
    const __m256i invWyVector = _mm256_set1_epi32(invWy);
    const __m256i rounding = _mm256_set1_epi32(kBilinearRounding);
    const __m128i zero = _mm_setzero_si128();

    for (; index + 8 <= count; index += 8) {
        const __m256i top = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row0 + index)
        );
        const __m256i bottom = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row1 + index)
        );
        __m256i value = _mm256_add_epi32(
            _mm256_mullo_epi32(top, invWyVector),
            _mm256_mullo_epi32(bottom, wyVector)
        );
        value = _mm256_add_epi32(value, rounding);
        value = _mm256_srai_epi32(value, 2 * kWeightBits);

        const __m128i lo = _mm256_castsi256_si128(value);
        const __m128i hi = _mm256_extracti128_si256(value, 1);
        const __m128i packed16 = _mm_packus_epi32(lo, hi);
        const __m128i packed8 = _mm_packus_epi16(packed16, zero);
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(outRow + index),
            packed8
        );
    }
#endif

    for (; index < count; ++index) {
        outRow[index] = intToU8(
            (row0[index] * invWy +
             row1[index] * wy +
             kBilinearRounding) >>
            (2 * kWeightBits)
        );
    }
}

void resizeRowsSimd(
    const std::uint8_t* src,
    std::uint8_t* dst,
    const ResizeConfig& config,
    const AxisTable& xTable,
    const AxisTable& yTable,
    int beginTask,
    int endTask
) {
    const std::size_t srcRowBytes =
        static_cast<std::size_t>(config.srcWidth) * 3;
    const std::size_t dstRowBytes =
        static_cast<std::size_t>(config.dstWidth) * 3;
    const std::size_t srcFrameBytes =
        static_cast<std::size_t>(config.srcHeight) * srcRowBytes;
    const std::size_t dstFrameBytes =
        static_cast<std::size_t>(config.dstHeight) * dstRowBytes;
    const int outputValues = config.dstWidth * 3;

    std::vector<int> scaledRow0(static_cast<std::size_t>(outputValues));
    std::vector<int> scaledRow1(static_cast<std::size_t>(outputValues));

    for (int task = beginTask; task < endTask; ++task) {
        const int frameIndex = task / config.dstHeight;
        const int dstY = task - frameIndex * config.dstHeight;

        const auto yIndex = static_cast<std::size_t>(dstY);
        const int y0 = yTable.lower[yIndex];
        const int y1 = yTable.upper[yIndex];
        const int wy = yTable.weightFixed[yIndex];
        const int invWy = kWeightScale - wy;

        const std::uint8_t* frameSrc =
            src + static_cast<std::size_t>(frameIndex) * srcFrameBytes;
        std::uint8_t* frameDst =
            dst + static_cast<std::size_t>(frameIndex) * dstFrameBytes;

        const std::uint8_t* row0 =
            frameSrc + static_cast<std::size_t>(y0) * srcRowBytes;
        std::uint8_t* outRow =
            frameDst + static_cast<std::size_t>(dstY) * dstRowBytes;

        horizontalFixedRow(row0, scaledRow0.data(), config, xTable);
        if (wy == 0 || y0 == y1) {
            storeHorizontalScaledU8(
                scaledRow0.data(),
                outRow,
                outputValues
            );
            continue;
        }

        const std::uint8_t* row1 =
            frameSrc + static_cast<std::size_t>(y1) * srcRowBytes;
        horizontalFixedRow(row1, scaledRow1.data(), config, xTable);
        storeVerticalMixedU8(
            scaledRow0.data(),
            scaledRow1.data(),
            outRow,
            outputValues,
            wy,
            invWy
        );
    }
}

class ResizeThreadPool {
public:
    explicit ResizeThreadPool(int threadCount)
        : threadCount_(threadCount) {
        workers_.reserve(static_cast<std::size_t>(threadCount_));
        for (int index = 0; index < threadCount_; ++index) {
            workers_.emplace_back(
                &ResizeThreadPool::workerLoop,
                this,
                index
            );
        }
    }

    ResizeThreadPool(const ResizeThreadPool&) = delete;
    ResizeThreadPool& operator=(const ResizeThreadPool&) = delete;

    ~ResizeThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        startCondition_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    int threadCount() const {
        return threadCount_;
    }

    void run(
        ResizeWorker worker,
        const std::uint8_t* src,
        std::uint8_t* dst,
        const ResizeConfig& config,
        const AxisTable& xTable,
        const AxisTable& yTable,
        int totalTasks
    ) {
        if (totalTasks <= 0) {
            return;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        worker_ = worker;
        src_ = src;
        dst_ = dst;
        config_ = &config;
        xTable_ = &xTable;
        yTable_ = &yTable;
        totalTasks_ = totalTasks;
        remainingWorkers_ = threadCount_;
        exception_ = nullptr;
        ++generation_;

        startCondition_.notify_all();
        doneCondition_.wait(lock, [&] {
            return remainingWorkers_ == 0;
        });

        if (exception_ != nullptr) {
            std::rethrow_exception(exception_);
        }
    }

private:
    void workerLoop(int workerIndex) {
        std::uint64_t seenGeneration = 0;

        for (;;) {
            ResizeWorker worker = nullptr;
            const std::uint8_t* src = nullptr;
            std::uint8_t* dst = nullptr;
            const ResizeConfig* config = nullptr;
            const AxisTable* xTable = nullptr;
            const AxisTable* yTable = nullptr;
            int totalTasks = 0;
            int threadCount = 0;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                startCondition_.wait(lock, [&] {
                    return stopping_ || generation_ != seenGeneration;
                });
                if (stopping_) {
                    return;
                }

                seenGeneration = generation_;
                worker = worker_;
                src = src_;
                dst = dst_;
                config = config_;
                xTable = xTable_;
                yTable = yTable_;
                totalTasks = totalTasks_;
                threadCount = threadCount_;
            }

            try {
                const int beginTask =
                    (totalTasks * workerIndex) / threadCount;
                const int endTask =
                    (totalTasks * (workerIndex + 1)) / threadCount;
                if (beginTask < endTask) {
                    worker(
                        src,
                        dst,
                        *config,
                        *xTable,
                        *yTable,
                        beginTask,
                        endTask
                    );
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (exception_ == nullptr) {
                    exception_ = std::current_exception();
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --remainingWorkers_;
                if (remainingWorkers_ == 0) {
                    doneCondition_.notify_one();
                }
            }
        }
    }

    int threadCount_ = 0;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable startCondition_;
    std::condition_variable doneCondition_;
    bool stopping_ = false;
    std::uint64_t generation_ = 0;
    int remainingWorkers_ = 0;
    ResizeWorker worker_ = nullptr;
    const std::uint8_t* src_ = nullptr;
    std::uint8_t* dst_ = nullptr;
    const ResizeConfig* config_ = nullptr;
    const AxisTable* xTable_ = nullptr;
    const AxisTable* yTable_ = nullptr;
    int totalTasks_ = 0;
    std::exception_ptr exception_ = nullptr;
};

std::mutex gThreadPoolMutex;
std::mutex gThreadPoolRunMutex;
std::unique_ptr<ResizeThreadPool> gThreadPool;

ResizeThreadPool& ensureThreadPool(int threadCount) {
    std::lock_guard<std::mutex> lock(gThreadPoolMutex);
    if (!gThreadPool ||
        gThreadPool->threadCount() != threadCount) {
        gThreadPool = std::make_unique<ResizeThreadPool>(threadCount);
    }
    return *gThreadPool;
}

int parseDsize(PyObject* dsizeObject, int& dstWidth, int& dstHeight) {
    PyObject* sequence = PySequence_Fast(
        dsizeObject,
        "dsize must be a sequence: (width, height)"
    );
    if (sequence == nullptr) {
        return -1;
    }

    if (PySequence_Fast_GET_SIZE(sequence) != 2) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "dsize must contain exactly two values");
        return -1;
    }

    PyObject** items = PySequence_Fast_ITEMS(sequence);
    const long widthValue = PyLong_AsLong(items[0]);
    if (PyErr_Occurred()) {
        Py_DECREF(sequence);
        return -1;
    }
    const long heightValue = PyLong_AsLong(items[1]);
    if (PyErr_Occurred()) {
        Py_DECREF(sequence);
        return -1;
    }
    Py_DECREF(sequence);

    if (widthValue <= 0 || heightValue <= 0) {
        PyErr_SetString(PyExc_ValueError, "dsize values must be positive");
        return -1;
    }

    dstWidth = static_cast<int>(widthValue);
    dstHeight = static_cast<int>(heightValue);
    return 0;
}

PyObject* resize(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* srcObject = nullptr;
    PyObject* dsizeObject = nullptr;
    int interpolation = 1;
    int requestedThreads = 0;

    static const char* keywords[] = {
        "src",
        "dsize",
        "interpolation",
        "threads",
        nullptr
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOi|i",
            const_cast<char**>(keywords),
            &srcObject,
            &dsizeObject,
            &interpolation,
            &requestedThreads)) {
        return nullptr;
    }

    if (interpolation != 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "fast_cpu_resize_simd.resize supports only cv2.INTER_LINEAR"
        );
        return nullptr;
    }
    if (requestedThreads < 0) {
        PyErr_SetString(PyExc_ValueError, "threads must be non-negative");
        return nullptr;
    }

    int dstWidth = 0;
    int dstHeight = 0;
    if (parseDsize(dsizeObject, dstWidth, dstHeight) != 0) {
        return nullptr;
    }

    PyArrayObject* srcArray = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(srcObject, NPY_UINT8, NPY_ARRAY_IN_ARRAY)
    );
    if (srcArray == nullptr) {
        return nullptr;
    }

    PyObject* resultObject = nullptr;

    try {
        if (!PyArray_ISCARRAY(srcArray)) {
            throw std::invalid_argument("src must be a C-contiguous uint8 ndarray");
        }

        const int ndim = PyArray_NDIM(srcArray);
        if (ndim != 3 && ndim != 4) {
            throw std::invalid_argument(
                "src must have shape (height, width, 3) or "
                "(batch, height, width, 3)"
            );
        }

        ResizeConfig config;
        config.batched = ndim == 4;
        config.batchSize = config.batched
            ? static_cast<int>(PyArray_DIM(srcArray, 0))
            : 1;
        config.srcHeight = static_cast<int>(
            PyArray_DIM(srcArray, config.batched ? 1 : 0)
        );
        config.srcWidth = static_cast<int>(
            PyArray_DIM(srcArray, config.batched ? 2 : 1)
        );
        const int channels = static_cast<int>(
            PyArray_DIM(srcArray, config.batched ? 3 : 2)
        );
        config.dstWidth = dstWidth;
        config.dstHeight = dstHeight;

        if (config.batchSize <= 0 ||
            config.srcWidth <= 0 ||
            config.srcHeight <= 0) {
            throw std::invalid_argument("src dimensions must be positive");
        }
        if (channels != 3) {
            throw std::invalid_argument("src must have exactly 3 channels");
        }

        npy_intp outputDims[4] = {
            static_cast<npy_intp>(config.batchSize),
            static_cast<npy_intp>(config.dstHeight),
            static_cast<npy_intp>(config.dstWidth),
            3
        };

        resultObject = PyArray_SimpleNew(
            config.batched ? 4 : 3,
            config.batched ? outputDims : outputDims + 1,
            NPY_UINT8
        );
        if (resultObject == nullptr) {
            Py_DECREF(srcArray);
            return nullptr;
        }

        const AxisTable xTable = makeAxisTable(config.srcWidth, config.dstWidth);
        const AxisTable yTable = makeAxisTable(config.srcHeight, config.dstHeight);
        const int totalTasks = config.batchSize * config.dstHeight;

        unsigned int hardwareThreads = std::thread::hardware_concurrency();
        if (hardwareThreads == 0) {
            hardwareThreads = 1;
        }
        const int threadCount = std::max(
            1,
            std::min(
                totalTasks,
                requestedThreads > 0
                    ? requestedThreads
                    : static_cast<int>(hardwareThreads)
            )
        );

        const auto* srcData = reinterpret_cast<const std::uint8_t*>(
            PyArray_DATA(srcArray)
        );
        auto* dstData = reinterpret_cast<std::uint8_t*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(resultObject))
        );

        PyThreadState* threadState = PyEval_SaveThread();
        try {
            if (threadCount == 1) {
                resizeRowsSimd(
                    srcData,
                    dstData,
                    config,
                    xTable,
                    yTable,
                    0,
                    totalTasks
                );
            } else {
                std::lock_guard<std::mutex> runLock(gThreadPoolRunMutex);
                ResizeThreadPool& threadPool = ensureThreadPool(threadCount);
                threadPool.run(
                    resizeRowsSimd,
                    srcData,
                    dstData,
                    config,
                    xTable,
                    yTable,
                    totalTasks
                );
            }
        } catch (...) {
            PyEval_RestoreThread(threadState);
            throw;
        }
        PyEval_RestoreThread(threadState);
    } catch (const std::exception& error) {
        Py_DECREF(srcArray);
        Py_XDECREF(resultObject);
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }

    Py_DECREF(srcArray);
    return resultObject;
}

PyMethodDef moduleMethods[] = {
    {
        "resize",
        reinterpret_cast<PyCFunction>(resize),
        METH_VARARGS | METH_KEYWORDS,
        "resize(src, dsize, interpolation, *, threads=0) -> numpy.ndarray"
    },
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef moduleDefinition = {
    PyModuleDef_HEAD_INIT,
    "fast_cpu_resize_simd",
    "SIMD-assisted fixed-point bilinear resize for uint8 HWC/NHWC RGB data.",
    -1,
    moduleMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit_fast_cpu_resize_simd() {
    if (_import_array() < 0) {
        return nullptr;
    }
    return PyModule_Create(&moduleDefinition);
}
