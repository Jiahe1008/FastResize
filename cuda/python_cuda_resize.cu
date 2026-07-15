#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_RESIZE_NO_MAIN
#include "cuda_resize.cu"

namespace {

struct PipelineKey {
    int srcWidth = 0;
    int srcHeight = 0;
    int dstWidth = 0;
    int dstHeight = 0;

    bool equals(int srcWidthValue,
                int srcHeightValue,
                int dstWidthValue,
                int dstHeightValue) const {
        return srcWidth == srcWidthValue &&
               srcHeight == srcHeightValue &&
               dstWidth == dstWidthValue &&
               dstHeight == dstHeightValue;
    }
};

std::mutex gPipelineMutex;
std::unique_ptr<CudaResizePipeline> gPipeline;
PipelineKey gPipelineKey;
bool gCudaInitialized = false;

struct RegisteredInput {
    void* pointer = nullptr;
    std::size_t bytes = 0;
    PyObject* owner = nullptr;
};

std::vector<RegisteredInput> gRegisteredInputs;

struct BatchResources {
    unsigned char* deviceSrc = nullptr;
    unsigned char* deviceDst = nullptr;
    std::size_t srcCapacity = 0;
    std::size_t dstCapacity = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t h2dStartEvent = nullptr;
    cudaEvent_t h2dEndEvent = nullptr;
    cudaEvent_t kernelEndEvent = nullptr;
    cudaEvent_t d2hEndEvent = nullptr;
};

BatchResources gBatchResources;

struct PinnedOutput {
    PyObject* array = nullptr;
    unsigned char* pointer = nullptr;
    std::size_t bytes = 0;
    int batchSize = 0;
    int dstWidth = 0;
    int dstHeight = 0;
    bool batched = false;
};

PinnedOutput gPinnedOutput;

void initializeCudaOnce() {
    if (gCudaInitialized) {
        return;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        throw std::runtime_error("No CUDA device is available");
    }

    CUDA_CHECK(cudaSetDevice(0));
    gCudaInitialized = true;
}

CudaResizePipeline& getPipeline(int srcWidth,
                                int srcHeight,
                                int dstWidth,
                                int dstHeight) {
    std::lock_guard<std::mutex> lock(gPipelineMutex);

    initializeCudaOnce();

    if (!gPipeline ||
        !gPipelineKey.equals(srcWidth, srcHeight, dstWidth, dstHeight)) {
        gPipeline = std::make_unique<CudaResizePipeline>(
            srcWidth,
            srcHeight,
            dstWidth,
            dstHeight
        );
        gPipelineKey = {srcWidth, srcHeight, dstWidth, dstHeight};
    }

    return *gPipeline;
}

void unregisterAllInputs() {
    for (RegisteredInput& input : gRegisteredInputs) {
        if (input.pointer != nullptr) {
            CUDA_CHECK(cudaHostUnregister(input.pointer));
            Py_XDECREF(input.owner);
        }
    }
    gRegisteredInputs.clear();
}

void releaseBatchResources() {
    if (gBatchResources.stream != nullptr) {
        cudaStreamSynchronize(gBatchResources.stream);
    }

    if (gBatchResources.h2dStartEvent != nullptr) {
        cudaEventDestroy(gBatchResources.h2dStartEvent);
        gBatchResources.h2dStartEvent = nullptr;
    }
    if (gBatchResources.h2dEndEvent != nullptr) {
        cudaEventDestroy(gBatchResources.h2dEndEvent);
        gBatchResources.h2dEndEvent = nullptr;
    }
    if (gBatchResources.kernelEndEvent != nullptr) {
        cudaEventDestroy(gBatchResources.kernelEndEvent);
        gBatchResources.kernelEndEvent = nullptr;
    }
    if (gBatchResources.d2hEndEvent != nullptr) {
        cudaEventDestroy(gBatchResources.d2hEndEvent);
        gBatchResources.d2hEndEvent = nullptr;
    }

    if (gBatchResources.stream != nullptr) {
        cudaStreamDestroy(gBatchResources.stream);
        gBatchResources.stream = nullptr;
    }
    if (gBatchResources.deviceSrc != nullptr) {
        cudaFree(gBatchResources.deviceSrc);
        gBatchResources.deviceSrc = nullptr;
    }
    if (gBatchResources.deviceDst != nullptr) {
        cudaFree(gBatchResources.deviceDst);
        gBatchResources.deviceDst = nullptr;
    }
    gBatchResources.srcCapacity = 0;
    gBatchResources.dstCapacity = 0;
}

void pinnedOutputCapsuleDestructor(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "cuda_pinned_output");
    if (pointer != nullptr) {
        cudaFreeHost(pointer);
    }
}

void releasePinnedOutput() {
    Py_XDECREF(gPinnedOutput.array);
    gPinnedOutput = {};
}

void ensureRegisteredInput(void* pointer, std::size_t bytes, PyObject* owner) {
    std::lock_guard<std::mutex> lock(gPipelineMutex);

    initializeCudaOnce();

    for (const RegisteredInput& input : gRegisteredInputs) {
        if (input.pointer == pointer && input.bytes == bytes) {
            return;
        }
    }

    CUDA_CHECK(cudaHostRegister(
        pointer,
        bytes,
        cudaHostRegisterDefault
    ));
    Py_INCREF(owner);
    gRegisteredInputs.push_back({pointer, bytes, owner});
}

BatchResources& ensureBatchResources(std::size_t srcBytes, std::size_t dstBytes) {
    std::lock_guard<std::mutex> lock(gPipelineMutex);

    initializeCudaOnce();

    if (gBatchResources.stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &gBatchResources.stream,
            cudaStreamNonBlocking
        ));
        CUDA_CHECK(cudaEventCreate(&gBatchResources.h2dStartEvent));
        CUDA_CHECK(cudaEventCreate(&gBatchResources.h2dEndEvent));
        CUDA_CHECK(cudaEventCreate(&gBatchResources.kernelEndEvent));
        CUDA_CHECK(cudaEventCreate(&gBatchResources.d2hEndEvent));
    }

    if (gBatchResources.srcCapacity < srcBytes) {
        if (gBatchResources.deviceSrc != nullptr) {
            CUDA_CHECK(cudaFree(gBatchResources.deviceSrc));
            gBatchResources.deviceSrc = nullptr;
            gBatchResources.srcCapacity = 0;
        }
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&gBatchResources.deviceSrc),
            srcBytes
        ));
        gBatchResources.srcCapacity = srcBytes;
    }

    if (gBatchResources.dstCapacity < dstBytes) {
        if (gBatchResources.deviceDst != nullptr) {
            CUDA_CHECK(cudaFree(gBatchResources.deviceDst));
            gBatchResources.deviceDst = nullptr;
            gBatchResources.dstCapacity = 0;
        }
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&gBatchResources.deviceDst),
            dstBytes
        ));
        gBatchResources.dstCapacity = dstBytes;
    }

    return gBatchResources;
}

PyArrayObject* ensurePinnedOutputArray(
    int batchSize,
    int dstWidth,
    int dstHeight,
    bool batched
) {
    const std::size_t dstBytes =
        static_cast<std::size_t>(batchSize) *
        static_cast<std::size_t>(dstHeight) *
        static_cast<std::size_t>(dstWidth) *
        3;

    if (gPinnedOutput.array != nullptr &&
        gPinnedOutput.bytes == dstBytes &&
        gPinnedOutput.batchSize == batchSize &&
        gPinnedOutput.dstWidth == dstWidth &&
        gPinnedOutput.dstHeight == dstHeight &&
        gPinnedOutput.batched == batched) {
        return reinterpret_cast<PyArrayObject*>(gPinnedOutput.array);
    }

    releasePinnedOutput();

    unsigned char* pointer = nullptr;
    CUDA_CHECK(cudaHostAlloc(
        reinterpret_cast<void**>(&pointer),
        dstBytes,
        cudaHostAllocDefault
    ));

    npy_intp outputDims[4] = {
        static_cast<npy_intp>(batchSize),
        static_cast<npy_intp>(dstHeight),
        static_cast<npy_intp>(dstWidth),
        3
    };

    PyObject* array = PyArray_SimpleNewFromData(
        batched ? 4 : 3,
        batched ? outputDims : outputDims + 1,
        NPY_UINT8,
        pointer
    );
    if (array == nullptr) {
        cudaFreeHost(pointer);
        return nullptr;
    }

    PyObject* capsule = PyCapsule_New(
        pointer,
        "cuda_pinned_output",
        pinnedOutputCapsuleDestructor
    );
    if (capsule == nullptr) {
        Py_DECREF(array);
        cudaFreeHost(pointer);
        return nullptr;
    }

    if (PyArray_SetBaseObject(
            reinterpret_cast<PyArrayObject*>(array),
            capsule
        ) != 0) {
        Py_DECREF(capsule);
        Py_DECREF(array);
        return nullptr;
    }

    gPinnedOutput.array = array;
    gPinnedOutput.pointer = pointer;
    gPinnedOutput.bytes = dstBytes;
    gPinnedOutput.batchSize = batchSize;
    gPinnedOutput.dstWidth = dstWidth;
    gPinnedOutput.dstHeight = dstHeight;
    gPinnedOutput.batched = batched;

    return reinterpret_cast<PyArrayObject*>(gPinnedOutput.array);
}

int parseDsize(PyObject* dsizeObject, int& dstWidth, int& dstHeight) {
    PyObject* sequence = PySequence_Fast(
        dsizeObject,
        "dsize must be a sequence: (width, height)"
    );
    if (sequence == nullptr) {return -1;}

    const Py_ssize_t length = PySequence_Fast_GET_SIZE(sequence);
    if (length != 2) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "dsize must contain exactly two values");
        return -1;
    }

    PyObject** items = PySequence_Fast_ITEMS(sequence);
    const long widthValue = PyLong_AsLong(items[0]);
    if (PyErr_Occurred()) { Py_DECREF(sequence); return -1;}

    const long heightValue = PyLong_AsLong(items[1]);
    if (PyErr_Occurred()) { Py_DECREF(sequence); return -1;}

    Py_DECREF(sequence);

    if (widthValue <= 0 || heightValue <= 0) {
        PyErr_SetString(PyExc_ValueError, "dsize values must be positive");
        return -1;
    }

    dstWidth = static_cast<int>(widthValue);
    dstHeight = static_cast<int>(heightValue);
    return 0;
}

PyObject* setPythonErrorFromException(const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
}

PyObject* profileToDict(
    const CudaResizePipeline::Profile& profile,
    int batchSize,
    double numpyOutputCopyMs
) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }

    const double batchSizeDouble = static_cast<double>(batchSize);
    auto setDouble = [&](const char* key, double value) -> bool {
        PyObject* object = PyFloat_FromDouble(value);
        if (object == nullptr) {
            return false;
        }
        const int result = PyDict_SetItemString(dict, key, object);
        Py_DECREF(object);
        return result == 0;
    };

    auto setPerFrame = [&](const char* key, double value) -> bool {
        std::string perFrameKey = std::string(key) + "_per_frame";
        return setDouble(perFrameKey.c_str(), value / batchSizeDouble);
    };

    if (!setDouble("pipeline_total_ms", profile.totalMs) ||
        !setDouble("input_copy_ms", profile.inputCopyMs) ||
        !setDouble("h2d_ms", profile.h2dMs) ||
        !setDouble("kernel_ms", profile.kernelMs) ||
        !setDouble("d2h_ms", profile.d2hMs) ||
        !setDouble("gpu_total_ms", profile.gpuTotalMs) ||
        !setDouble("output_copy_ms", profile.outputCopyMs) ||
        !setDouble("numpy_output_copy_ms", numpyOutputCopyMs) ||
        !setDouble("total_with_numpy_copy_ms",
                   profile.totalMs + numpyOutputCopyMs) ||
        !setPerFrame("pipeline_total_ms", profile.totalMs) ||
        !setPerFrame("input_copy_ms", profile.inputCopyMs) ||
        !setPerFrame("h2d_ms", profile.h2dMs) ||
        !setPerFrame("kernel_ms", profile.kernelMs) ||
        !setPerFrame("d2h_ms", profile.d2hMs) ||
        !setPerFrame("gpu_total_ms", profile.gpuTotalMs) ||
        !setPerFrame("output_copy_ms", profile.outputCopyMs) ||
        !setPerFrame("numpy_output_copy_ms", numpyOutputCopyMs) ||
        !setPerFrame("total_with_numpy_copy_ms",
                     profile.totalMs + numpyOutputCopyMs)) {
        Py_DECREF(dict);
        return nullptr;
    }

    PyObject* batchObject = PyLong_FromLong(batchSize);
    if (batchObject == nullptr) {
        Py_DECREF(dict);
        return nullptr;
    }
    const int batchResult = PyDict_SetItemString(dict, "batch_size", batchObject);
    Py_DECREF(batchObject);
    if (batchResult != 0) {
        Py_DECREF(dict);
        return nullptr;
    }

    return dict;
}

void runContiguousBatchResize(
    const unsigned char* srcData,
    unsigned char* dstData,
    int batchSize,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    CudaResizePipeline::Profile* profile
) {
    const std::size_t srcRowBytes =
        static_cast<std::size_t>(srcWidth) * 3;
    const std::size_t dstRowBytes =
        static_cast<std::size_t>(dstWidth) * 3;
    const std::size_t srcFrameBytes =
        static_cast<std::size_t>(srcHeight) * srcRowBytes;
    const std::size_t dstFrameBytes =
        static_cast<std::size_t>(dstHeight) * dstRowBytes;
    const std::size_t srcBytes =
        static_cast<std::size_t>(batchSize) * srcFrameBytes;
    const std::size_t dstBytes =
        static_cast<std::size_t>(batchSize) * dstFrameBytes;

    const auto totalBegin = std::chrono::steady_clock::now();

    BatchResources& resources = ensureBatchResources(srcBytes, dstBytes);

    if (profile != nullptr) {
        CUDA_CHECK(cudaEventRecord(
            resources.h2dStartEvent,
            resources.stream
        ));
    }

    CUDA_CHECK(cudaMemcpyAsync(
        resources.deviceSrc,
        srcData,
        srcBytes,
        cudaMemcpyHostToDevice,
        resources.stream
    ));

    if (profile != nullptr) {
        CUDA_CHECK(cudaEventRecord(
            resources.h2dEndEvent,
            resources.stream
        ));
    }

    const dim3 block(32, 8);
    const dim3 grid(
        static_cast<unsigned int>((dstWidth + block.x - 1) / block.x),
        static_cast<unsigned int>((dstHeight + block.y - 1) / block.y)
    );

    for (int index = 0; index < batchSize; ++index) {
        resizeBilinearBgr8Kernel<<<
            grid,
            block,
            0,
            resources.stream
        >>>(
            resources.deviceSrc +
                static_cast<std::size_t>(index) * srcFrameBytes,
            srcRowBytes,
            srcWidth,
            srcHeight,
            resources.deviceDst +
                static_cast<std::size_t>(index) * dstFrameBytes,
            dstRowBytes,
            dstWidth,
            dstHeight
        );
    }
    CUDA_CHECK(cudaGetLastError());

    if (profile != nullptr) {
        CUDA_CHECK(cudaEventRecord(
            resources.kernelEndEvent,
            resources.stream
        ));
    }

    CUDA_CHECK(cudaMemcpyAsync(
        dstData,
        resources.deviceDst,
        dstBytes,
        cudaMemcpyDeviceToHost,
        resources.stream
    ));

    if (profile != nullptr) {
        CUDA_CHECK(cudaEventRecord(
            resources.d2hEndEvent,
            resources.stream
        ));
    }

    CUDA_CHECK(cudaStreamSynchronize(resources.stream));

    const auto totalEnd = std::chrono::steady_clock::now();

    if (profile != nullptr) {
        float h2dMs = 0.0f;
        float kernelMs = 0.0f;
        float d2hMs = 0.0f;
        float gpuTotalMs = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &h2dMs,
            resources.h2dStartEvent,
            resources.h2dEndEvent
        ));
        CUDA_CHECK(cudaEventElapsedTime(
            &kernelMs,
            resources.h2dEndEvent,
            resources.kernelEndEvent
        ));
        CUDA_CHECK(cudaEventElapsedTime(
            &d2hMs,
            resources.kernelEndEvent,
            resources.d2hEndEvent
        ));
        CUDA_CHECK(cudaEventElapsedTime(
            &gpuTotalMs,
            resources.h2dStartEvent,
            resources.d2hEndEvent
        ));

        profile->totalMs =
            std::chrono::duration<double, std::milli>(
                totalEnd - totalBegin
            ).count();
        profile->inputCopyMs = 0.0;
        profile->outputCopyMs = 0.0;
        profile->h2dMs = h2dMs;
        profile->kernelMs = kernelMs;
        profile->d2hMs = d2hMs;
        profile->gpuTotalMs = gpuTotalMs;
    }
}

PyObject* resizeImpl(PyObject* args, bool withProfile) {
    PyObject* srcObject = nullptr;
    PyObject* dsizeObject = nullptr;
    int interpolation = cv::INTER_LINEAR;

    if (!PyArg_ParseTuple(args, "OOi", &srcObject, &dsizeObject, &interpolation)) {
        return nullptr;
    }

    if (interpolation != cv::INTER_LINEAR) {
        PyErr_SetString(
            PyExc_ValueError,
            "cuda_resize_py.resize currently supports only cv2.INTER_LINEAR"
        );
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
    if (reinterpret_cast<PyObject*>(srcArray) != srcObject) {
        Py_DECREF(srcArray);
        PyErr_SetString(
            PyExc_ValueError,
            "src must be a uint8 C-contiguous numpy.ndarray; "
            "implicit conversion/copy is not allowed for cudaHostRegister"
        );
        return nullptr;
    }

    PyObject* resultObject = nullptr;
    PyObject* profileObject = nullptr;

    try {
        const int ndim = PyArray_NDIM(srcArray);
        if (ndim != 3 && ndim != 4) {
            throw std::invalid_argument(
                "src must have shape (height, width, 3) or "
                "(batch, height, width, 3)"
            );
        }

        const bool batched = ndim == 4;
        const int batchSize = batched
            ? static_cast<int>(PyArray_DIM(srcArray, 0))
            : 1;
        const int srcHeight = static_cast<int>(PyArray_DIM(srcArray, batched ? 1 : 0));
        const int srcWidth = static_cast<int>(PyArray_DIM(srcArray, batched ? 2 : 1));
        const int channels = static_cast<int>(PyArray_DIM(srcArray, batched ? 3 : 2));

        if (batchSize <= 0 || srcWidth <= 0 || srcHeight <= 0) {
            throw std::invalid_argument("src dimensions must be positive");
        }
        if (channels != 3) {
            throw std::invalid_argument("src must have exactly 3 channels");
        }

        unsigned char* srcData = reinterpret_cast<unsigned char*>(
            PyArray_DATA(srcArray)
        );
        if (!PyArray_ISCARRAY(srcArray)) {
            throw std::invalid_argument("src must be a C-contiguous uint8 ndarray");
        }

        const std::size_t srcArrayBytes =
            static_cast<std::size_t>(PyArray_NBYTES(srcArray));
        ensureRegisteredInput(
            srcData,
            srcArrayBytes,
            reinterpret_cast<PyObject*>(srcArray)
        );

        auto* resultArray = ensurePinnedOutputArray(
            batchSize,
            dstWidth,
            dstHeight,
            batched
        );
        if (resultArray == nullptr) {
            Py_DECREF(srcArray);
            return nullptr;
        }
        resultObject = reinterpret_cast<PyObject*>(resultArray);
        Py_INCREF(resultObject);

        unsigned char* resultData = reinterpret_cast<unsigned char*>(
            PyArray_DATA(resultArray)
        );

        CudaResizePipeline::Profile profile;

        PyThreadState* threadState = PyEval_SaveThread();
        try {
            runContiguousBatchResize(
                srcData,
                resultData,
                batchSize,
                srcWidth,
                srcHeight,
                dstWidth,
                dstHeight,
                withProfile ? &profile : nullptr
            );
        } catch (...) {
            PyEval_RestoreThread(threadState);
            throw;
        }
        PyEval_RestoreThread(threadState);

        if (withProfile) {
            profileObject = profileToDict(
                profile,
                batchSize,
                0.0
            );
            if (profileObject == nullptr) {
                Py_DECREF(srcArray);
                Py_DECREF(resultObject);
                return nullptr;
            }
        }
    } catch (const cv::Exception& error) {
        Py_DECREF(srcArray);
        Py_XDECREF(resultObject);
        Py_XDECREF(profileObject);
        return setPythonErrorFromException(error);
    } catch (const std::exception& error) {
        Py_DECREF(srcArray);
        Py_XDECREF(resultObject);
        Py_XDECREF(profileObject);
        return setPythonErrorFromException(error);
    }

    Py_DECREF(srcArray);
    if (withProfile) {
        PyObject* tuple = PyTuple_New(2);
        if (tuple == nullptr) {
            Py_DECREF(resultObject);
            Py_DECREF(profileObject);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, 0, resultObject);
        PyTuple_SET_ITEM(tuple, 1, profileObject);
        return tuple;
    }
    return resultObject;
}

PyObject* resize(PyObject*, PyObject* args) {
    return resizeImpl(args, false);
}

PyObject* resizeProfile(PyObject*, PyObject* args) {
    return resizeImpl(args, true);
}

PyObject* synchronize(PyObject*, PyObject*) {
    try {
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& error) {
        return setPythonErrorFromException(error);
    }

    Py_RETURN_NONE;
}

PyObject* clearRegisteredInputs(PyObject*, PyObject*) {
    try {
        std::lock_guard<std::mutex> lock(gPipelineMutex);
        unregisterAllInputs();
    } catch (const std::exception& error) {
        return setPythonErrorFromException(error);
    }

    Py_RETURN_NONE;
}

PyMethodDef moduleMethods[] = {
    {
        "resize",
        resize,
        METH_VARARGS,
        "resize(src, dsize, interpolation) -> numpy.ndarray"
    },
    {
        "resize_profile",
        resizeProfile,
        METH_VARARGS,
        "resize_profile(src, dsize, interpolation) -> (numpy.ndarray, dict)"
    },
    {
        "synchronize",
        synchronize,
        METH_NOARGS,
        "Synchronize the current CUDA device."
    },
    {
        "clear_registered_inputs",
        clearRegisteredInputs,
        METH_NOARGS,
        "Unregister all cached host input pointers."
    },
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef moduleDefinition = {
    PyModuleDef_HEAD_INIT,
    "cuda_resize_py",
    "CUDA bilinear BGR8 resize module.",
    -1,
    moduleMethods,
    nullptr,
    nullptr,
    nullptr,
    [](void*) {
        try {
            unregisterAllInputs();
            releaseBatchResources();
            releasePinnedOutput();
        } catch (...) {
        }
    }
};

}  // namespace

PyMODINIT_FUNC PyInit_cuda_resize_py() {
    if (_import_array() < 0) {
        return nullptr;
    }

    return PyModule_Create(&moduleDefinition);
}
