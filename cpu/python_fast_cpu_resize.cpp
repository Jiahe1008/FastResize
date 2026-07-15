#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kWeightBits = 11;
constexpr int kWeightScale = 1 << kWeightBits;
constexpr int kBilinearRounding = 1 << (2 * kWeightBits - 1);

struct AxisTable {
    std::vector<int> lower;
    std::vector<int> upper;
    std::vector<float> weight;
    std::vector<int> weightFixed;
};

struct ResizeConfig {
    int batchSize = 1;
    int srcWidth = 0;
    int srcHeight = 0;
    int dstWidth = 0;
    int dstHeight = 0;
    bool batched = false;
};

AxisTable makeAxisTable(int srcSize, int dstSize) {
    AxisTable table;
    table.lower.resize(static_cast<std::size_t>(dstSize));
    table.upper.resize(static_cast<std::size_t>(dstSize));
    table.weight.resize(static_cast<std::size_t>(dstSize));
    table.weightFixed.resize(static_cast<std::size_t>(dstSize));

    const float scale =
        static_cast<float>(srcSize) / static_cast<float>(dstSize);

    for (int dst = 0; dst < dstSize; ++dst) {
        const float src = (static_cast<float>(dst) + 0.5f) * scale - 0.5f;
        const int rawLower = static_cast<int>(std::floor(src));
        const int rawUpper = rawLower + 1;
        table.lower[static_cast<std::size_t>(dst)] =
            std::clamp(rawLower, 0, srcSize - 1);
        table.upper[static_cast<std::size_t>(dst)] =
            std::clamp(rawUpper, 0, srcSize - 1);
        const float weight = src - static_cast<float>(rawLower);
        table.weight[static_cast<std::size_t>(dst)] = weight;
        table.weightFixed[static_cast<std::size_t>(dst)] = std::clamp(
            static_cast<int>(std::lrintf(weight * kWeightScale)),
            0,
            kWeightScale
        );
    }

    return table;
}

inline std::uint8_t floatToU8(float value) {
    const int rounded = static_cast<int>(std::lrintf(value));
    return static_cast<std::uint8_t>(std::clamp(rounded, 0, 255));
}

inline std::uint8_t intToU8(int value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

[[maybe_unused]] void resizeRowsFloatReference(
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

    for (int task = beginTask; task < endTask; ++task) {
        const int frameIndex = task / config.dstHeight;
        const int dstY = task - frameIndex * config.dstHeight;

        const auto yIndex = static_cast<std::size_t>(dstY);
        const int y0 = yTable.lower[yIndex];
        const int y1 = yTable.upper[yIndex];
        const float wy = yTable.weight[yIndex];

        const std::uint8_t* frameSrc =
            src + static_cast<std::size_t>(frameIndex) * srcFrameBytes;
        std::uint8_t* frameDst =
            dst + static_cast<std::size_t>(frameIndex) * dstFrameBytes;

        const std::uint8_t* row0 =
            frameSrc + static_cast<std::size_t>(y0) * srcRowBytes;
        const std::uint8_t* row1 =
            frameSrc + static_cast<std::size_t>(y1) * srcRowBytes;
        std::uint8_t* outRow =
            frameDst + static_cast<std::size_t>(dstY) * dstRowBytes;

        for (int dstX = 0; dstX < config.dstWidth; ++dstX) {
            const auto xIndex = static_cast<std::size_t>(dstX);
            const int x0 = xTable.lower[xIndex];
            const int x1 = xTable.upper[xIndex];
            const float wx = xTable.weight[xIndex];

            const int offset0 = x0 * 3;
            const int offset1 = x1 * 3;
            const int dstOffset = dstX * 3;

            for (int channel = 0; channel < 3; ++channel) {
                const float p00 =
                    static_cast<float>(row0[offset0 + channel]);
                const float p01 =
                    static_cast<float>(row0[offset1 + channel]);
                const float p10 =
                    static_cast<float>(row1[offset0 + channel]);
                const float p11 =
                    static_cast<float>(row1[offset1 + channel]);
                const float top = p00 + wx * (p01 - p00);
                const float bottom = p10 + wx * (p11 - p10);
                const float value = top + wy * (bottom - top);
                outRow[dstOffset + channel] = floatToU8(value);
            }
        }
    }
}

inline std::uint8_t bilinearFixedU8(
    int p00,
    int p01,
    int p10,
    int p11,
    int wx,
    int invWx,
    int wy,
    int invWy
) {
    const int top = p00 * invWx + p01 * wx;
    const int bottom = p10 * invWx + p11 * wx;
    const int value =
        (top * invWy + bottom * wy + kBilinearRounding) >>
        (2 * kWeightBits);
    return intToU8(value);
}

inline std::uint8_t horizontalFixedU8(
    int p0,
    int p1,
    int wx,
    int invWx
) {
    const int value =
        (p0 * invWx + p1 * wx + (1 << (kWeightBits - 1))) >>
        kWeightBits;
    return intToU8(value);
}

void resizeRows(
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

        if (wy == 0 || y0 == y1) {
            for (int dstX = 0; dstX < config.dstWidth; ++dstX) {
                const auto xIndex = static_cast<std::size_t>(dstX);
                const int x0 = xTable.lower[xIndex];
                const int x1 = xTable.upper[xIndex];
                const int wx = xTable.weightFixed[xIndex];
                const int invWx = kWeightScale - wx;

                const int offset0 = x0 * 3;
                const int offset1 = x1 * 3;
                const int dstOffset = dstX * 3;

                outRow[dstOffset] = horizontalFixedU8(
                    row0[offset0],
                    row0[offset1],
                    wx,
                    invWx
                );
                outRow[dstOffset + 1] = horizontalFixedU8(
                    row0[offset0 + 1],
                    row0[offset1 + 1],
                    wx,
                    invWx
                );
                outRow[dstOffset + 2] = horizontalFixedU8(
                    row0[offset0 + 2],
                    row0[offset1 + 2],
                    wx,
                    invWx
                );
            }
            continue;
        }

        const std::uint8_t* row1 =
            frameSrc + static_cast<std::size_t>(y1) * srcRowBytes;

        for (int dstX = 0; dstX < config.dstWidth; ++dstX) {
            const auto xIndex = static_cast<std::size_t>(dstX);
            const int x0 = xTable.lower[xIndex];
            const int x1 = xTable.upper[xIndex];
            const int wx = xTable.weightFixed[xIndex];
            const int invWx = kWeightScale - wx;

            const int offset0 = x0 * 3;
            const int offset1 = x1 * 3;
            const int dstOffset = dstX * 3;

            outRow[dstOffset] = bilinearFixedU8(
                row0[offset0],
                row0[offset1],
                row1[offset0],
                row1[offset1],
                wx,
                invWx,
                wy,
                invWy
            );
            outRow[dstOffset + 1] = bilinearFixedU8(
                row0[offset0 + 1],
                row0[offset1 + 1],
                row1[offset0 + 1],
                row1[offset1 + 1],
                wx,
                invWx,
                wy,
                invWy
            );
            outRow[dstOffset + 2] = bilinearFixedU8(
                row0[offset0 + 2],
                row0[offset1 + 2],
                row1[offset0 + 2],
                row1[offset1 + 2],
                wx,
                invWx,
                wy,
                invWy
            );
        }
    }
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

PyObject* resizeWithKernel(PyObject* args, PyObject* kwargs, bool useFixedPoint) {
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
            "fast_cpu_resize.resize currently supports only cv2.INTER_LINEAR"
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
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(threadCount));

            for (int index = 0; index < threadCount; ++index) {
                const int beginTask =
                    (totalTasks * index) / threadCount;
                const int endTask =
                    (totalTasks * (index + 1)) / threadCount;
                workers.emplace_back(
                    useFixedPoint ? resizeRows : resizeRowsFloatReference,
                    srcData,
                    dstData,
                    std::cref(config),
                    std::cref(xTable),
                    std::cref(yTable),
                    beginTask,
                    endTask
                );
            }

            for (std::thread& worker : workers) {
                worker.join();
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

PyObject* resize(PyObject*, PyObject* args, PyObject* kwargs) {
    return resizeWithKernel(args, kwargs, true);
}

PyObject* resizeFixed(PyObject*, PyObject* args, PyObject* kwargs) {
    return resizeWithKernel(args, kwargs, true);
}

PyObject* resizeFloat(PyObject*, PyObject* args, PyObject* kwargs) {
    return resizeWithKernel(args, kwargs, false);
}

PyMethodDef moduleMethods[] = {
    {
        "resize",
        reinterpret_cast<PyCFunction>(resize),
        METH_VARARGS | METH_KEYWORDS,
        "resize(src, dsize, interpolation, *, threads=0) -> numpy.ndarray"
    },
    {
        "resize_fixed",
        reinterpret_cast<PyCFunction>(resizeFixed),
        METH_VARARGS | METH_KEYWORDS,
        "resize_fixed(src, dsize, interpolation, *, threads=0) -> numpy.ndarray"
    },
    {
        "resize_float",
        reinterpret_cast<PyCFunction>(resizeFloat),
        METH_VARARGS | METH_KEYWORDS,
        "resize_float(src, dsize, interpolation, *, threads=0) -> numpy.ndarray"
    },
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef moduleDefinition = {
    PyModuleDef_HEAD_INIT,
    "fast_cpu_resize",
    "Generic multithreaded bilinear resize for uint8 HWC/NHWC RGB data.",
    -1,
    moduleMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit_fast_cpu_resize() {
    if (_import_array() < 0) {
        return nullptr;
    }
    return PyModule_Create(&moduleDefinition);
}
