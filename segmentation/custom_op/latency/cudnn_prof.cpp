#include <thread>
#include <cassert>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cudnn.h>
#include <chrono>

static int
checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code, cudaGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

static int
checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

#define checkCudaErr(...)                                                        \
    do {                                                                         \
        int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err) {                                                               \
            numErrors++;                                                         \
            goto clean;                                                          \
        }                                                                        \
    } while (0)

#define checkCudnnErr(...)                                                        \
    do {                                                                          \
        int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err) {                                                                \
            numErrors++;                                                          \
            goto clean;                                                           \
        }                                                                         \
    } while (0)

#define duration_ms(start, end) std::chrono::duration_cast<std::chrono::microseconds>((end) - (start)).count() / 1000.0
#define to_us(t) std::chrono::duration_cast<std::chrono::microseconds>(t)

static void
generateStrides(const int* dimA, int* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
    if (filterFormat == CUDNN_TENSOR_NCHW || filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
        strideA[nbDims - 1] = 1;
        for (int d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1]          = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}

struct ProfileResult {
    std::chrono::microseconds forward_start_timestamp;
    std::chrono::microseconds forward_end_timestamp;
    std::chrono::microseconds backward_start_timestamp;
    std::chrono::microseconds backward_end_timestamp;
    float forward_time;
    float backward_time;
    size_t context_size;
};

void print_profiling_result(ProfileResult &result) {
    float fwd_time, bwd_time;
    printf("  Forward Time: %.3fms\n", result.forward_time);
    printf("  Backward Time: %.3fms\n", result.backward_time);
    printf("  Total Time: %.3fms\n", result.forward_time + result.backward_time);
    printf("  Context Size: %.3fKB\n", result.context_size / 1024.0);
}

int convFwd(cudnnHandle_t handle_,
            float* devPtrX,
            float* devPtrW,
            float* devPtrY,
            cudnnTensorDescriptor_t cudnnXdesc,
            cudnnFilterDescriptor_t cudnnWdesc,
            cudnnTensorDescriptor_t cudnnYdesc,
            cudnnConvolutionDescriptor_t cudnnConvDesc,
            float alpha, float beta, int ySize, bool sync
) {
    void* workSpace = 0;
    size_t workSpaceSize;
    int numErrors = 0;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    checkCudnnErr(cudnnGetConvolutionForwardWorkspaceSize(
        handle_, cudnnXdesc, cudnnWdesc, cudnnConvDesc, cudnnYdesc, algo, &workSpaceSize));
    if (workSpaceSize > 0) {
        checkCudaErr(cudaMalloc(&workSpace, workSpaceSize));
    }
    checkCudnnErr(cudnnConvolutionForward(handle_,
                                          (void*)(&alpha),
                                          cudnnXdesc,
                                          devPtrX,
                                          cudnnWdesc,
                                          devPtrW,
                                          cudnnConvDesc,
                                          algo,
                                          workSpace,
                                          workSpaceSize,
                                          (void*)(&beta),
                                          cudnnYdesc,
                                          devPtrY));
    if (sync) checkCudaErr(cudaDeviceSynchronize());
clean:
    if (workSpace) cudaFree(workSpace);
    return numErrors;
}


int convBwdW(cudnnHandle_t handle_,
             float* devPtrX,
             float* devPtrW,
             float* devPtrY,
             cudnnTensorDescriptor_t cudnnXdesc,
             cudnnFilterDescriptor_t cudnnWdesc,
             cudnnTensorDescriptor_t cudnnYdesc,
             cudnnConvolutionDescriptor_t cudnnConvDesc,
             float alpha, float beta, int wSize, bool sync
                  ) {
    cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

    void* workSpace = 0;
    size_t workSpaceSize;
    int numErrors = 0;
    checkCudnnErr(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle_, cudnnXdesc, cudnnYdesc, cudnnConvDesc, cudnnWdesc, algo, &workSpaceSize));

    if (workSpaceSize > 0) {
        checkCudaErr(cudaMalloc(&workSpace, workSpaceSize));
    }
    checkCudnnErr(cudnnConvolutionBackwardFilter(handle_,
                                                 (void*)(&alpha),
                                                 cudnnXdesc,
                                                 devPtrX,
                                                 cudnnYdesc,
                                                 devPtrY,
                                                 cudnnConvDesc,
                                                 algo,
                                                 workSpace,
                                                 workSpaceSize,
                                                 (void*)(&beta),
                                                 cudnnWdesc,
                                                 devPtrW));
    if (sync) checkCudaErr(cudaDeviceSynchronize());
    
clean:
    if (workSpace) cudaFree(workSpace);
    return numErrors;
}

int convBwdX(cudnnHandle_t handle_,
             float* devPtrX,
             float* devPtrW,
             float* devPtrY,
             cudnnTensorDescriptor_t cudnnXdesc,
             cudnnFilterDescriptor_t cudnnWdesc,
             cudnnTensorDescriptor_t cudnnYdesc,
             cudnnConvolutionDescriptor_t cudnnConvDesc,
             float alpha, float beta, int xSize, bool sync
                  ) {
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    void* workSpace = 0;
    size_t workSpaceSize;
    int numErrors = 0;
    checkCudnnErr(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle_, cudnnWdesc, cudnnYdesc, cudnnConvDesc, cudnnXdesc, algo, &workSpaceSize));
    if (workSpaceSize > 0) {
        checkCudaErr(cudaMalloc(&workSpace, workSpaceSize));
    }

    checkCudnnErr(cudnnConvolutionBackwardData(handle_,
                                               (void*)(&alpha),
                                               cudnnWdesc,
                                               devPtrW,
                                               cudnnYdesc,
                                               devPtrY,
                                               cudnnConvDesc,
                                               algo,
                                               workSpace,
                                               workSpaceSize,
                                               (void*)(&beta),
                                               cudnnXdesc,
                                               devPtrX));

    if (sync) checkCudaErr(cudaDeviceSynchronize());
clean:
    if (workSpace) cudaFree(workSpace);
    return numErrors;
}

int avgConv(cudnnHandle_t handle_, 
            int xShape[], int wShape[], int yShape[], int convStride[],
            cudnnTensorDescriptor_t xDesc,
            cudnnFilterDescriptor_t rawWdesc,
            cudnnTensorDescriptor_t yDesc,
            float* devPtrX,
            float* devPtrY,
            float* devPtrW,
            int wStride[],
            int avgRad, int repeat, ProfileResult &result
            ) {
    int numErrors = 0;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    float contextTime, filterTime, gradXtime, gradWtime;
    cudnnTensorDescriptor_t wDesc;

    const int gradFilterSize[2] = {avgRad, avgRad};
    int xPoolingSize[2];
    int xPad[2], gradYpad[2];
    int avgGradYshape[4], avgXshape[4], sumWshape[4];
    int tempShape[4];
    int ySize = yShape[0] * yShape[1] * yShape[2] * yShape[3];
    int fwdConvPad[] = {1, 1}, fwdConvDilation[] = {1, 1};

    cudnnTensorDescriptor_t avgGradYdesc;
    cudnnTensorDescriptor_t avgXdesc;
    cudnnTensorDescriptor_t sumWdesc;
    cudnnFilterDescriptor_t sumWfiltDesc;
    int avgGradYstride[4], avgXstride[4], sumWstride[4];
    int avgGradYsize = 1, avgXsize = 1, sumWsize = 1;

    float* devAvgX;
    float* devAvgGradY;
    float* devSumW;

    cudnnReduceTensorDescriptor_t wReduceDesc;
    cudnnPoolingDescriptor_t gradYpoolingDesc;
    cudnnPoolingDescriptor_t xPoolingDesc;
    cudnnConvolutionDescriptor_t gradY2gradXConvDesc;
    cudnnConvolutionDescriptor_t cudnnConvDesc;

    cudnnCreateConvolutionDescriptor(&cudnnConvDesc);
    cudnnCreateTensorDescriptor(&wDesc);
    cudnnSetTensorNdDescriptor(wDesc, CUDNN_DATA_FLOAT, 4, wShape, wStride);

    for (int dim = 0; dim < 2; dim++) {
        xPoolingSize[dim] = gradFilterSize[dim] * convStride[dim];
        avgXshape[dim] = xShape[dim];
        avgXshape[dim + 2] = std::ceil(xShape[2 + dim] * 1.0 / xPoolingSize[dim]);
        xPad[dim] = std::ceil((avgXshape[dim + 2] * xPoolingSize[dim] - xShape[dim + 2]) / 2.0);
        avgGradYshape[dim] = yShape[dim];
        avgGradYshape[dim + 2] = std::ceil(yShape[2 + dim] * 1.0 / gradFilterSize[dim]);
        gradYpad[dim] = std::ceil((avgGradYshape[dim + 2] * gradFilterSize[dim] - yShape[dim + 2]) / 2.0);
        sumWshape[dim] = wShape[dim];
        sumWshape[dim + 2] = 1;
    }
    cudnnCreateTensorDescriptor(&avgGradYdesc);
    cudnnCreateTensorDescriptor(&avgXdesc);
    cudnnCreateTensorDescriptor(&sumWdesc);
    cudnnCreateFilterDescriptor(&sumWfiltDesc);
    generateStrides(avgXshape, avgXstride, 4, CUDNN_TENSOR_NCHW);
    generateStrides(avgGradYshape, avgGradYstride, 4, CUDNN_TENSOR_NCHW);
    generateStrides(sumWshape, sumWstride, 4, CUDNN_TENSOR_NCHW);
    for (int dim = 0; dim < 4; dim++) {
        avgXsize *= avgXshape[dim];
        avgGradYsize *= avgGradYshape[dim];
        sumWsize *= sumWshape[dim];
    }
    cudnnSetConvolutionNdDescriptor(
        cudnnConvDesc, 2, fwdConvPad, convStride, fwdConvDilation, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    cudnnSetTensorNdDescriptor(avgGradYdesc, CUDNN_DATA_FLOAT, 4, avgGradYshape, avgGradYstride);
    cudnnSetTensorNdDescriptor(avgXdesc, CUDNN_DATA_FLOAT, 4, avgXshape, avgXstride);
    cudnnSetTensorNdDescriptor(sumWdesc, CUDNN_DATA_FLOAT, 4, sumWshape, sumWstride);
    cudnnSetFilterNdDescriptor(sumWfiltDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, sumWshape);
    cudaMalloc((void**)&(devAvgX), avgXsize * sizeof(devAvgX[0]));
    cudaMalloc((void**)&(devAvgGradY), avgGradYsize * sizeof(devAvgGradY[0]));
    cudaMalloc((void**)&(devSumW), sumWsize * sizeof(devSumW[0]));

    cudnnCreateReduceTensorDescriptor(&wReduceDesc);
    cudnnSetReduceTensorDescriptor(wReduceDesc, 
                                   cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD, 
                                   cudnnDataType_t::CUDNN_DATA_FLOAT, 
                                   cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN, 
                                   cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_NO_INDICES, 
                                   cudnnIndicesType_t::CUDNN_32BIT_INDICES);

    cudnnCreatePoolingDescriptor(&gradYpoolingDesc);
    cudnnSetPooling2dDescriptor(gradYpoolingDesc, 
                                cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 
                                cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN, 
                                gradFilterSize[0], gradFilterSize[1], 
                                gradYpad[0], gradYpad[1], gradFilterSize[0], gradFilterSize[1]);
    cudnnGetPooling2dForwardOutputDim(gradYpoolingDesc, yDesc, 
                                      &tempShape[0], &tempShape[1], &tempShape[2], &tempShape[3]);
    for (int dim = 0; dim < 4; dim++) {
        assert(tempShape[dim] == avgGradYshape[dim]);
    }

    cudnnCreatePoolingDescriptor(&xPoolingDesc);
    cudnnSetPooling2dDescriptor(xPoolingDesc,
                                cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN, 
                                xPoolingSize[0], xPoolingSize[1], 
                                xPad[0], xPad[1], xPoolingSize[0], xPoolingSize[1]);
    cudnnGetPooling2dForwardOutputDim(xPoolingDesc, xDesc, 
                                      &tempShape[0], &tempShape[1], &tempShape[2], &tempShape[3]);
    for (int dim = 0; dim < 4; dim++) {
        assert(tempShape[dim] == avgXshape[dim]);
    }

    cudnnCreateConvolutionDescriptor(&gradY2gradXConvDesc);
    cudnnSetConvolution2dDescriptor(gradY2gradXConvDesc, 0, 0, convStride[0], convStride[1], 1, 1, 
                                    cudnnConvolutionMode_t::CUDNN_CONVOLUTION, 
                                    cudnnDataType_t::CUDNN_DATA_FLOAT);

    void* workSpace = nullptr;
    size_t workSpaceSize;
    float alpha = 1.0, beta = 0.0;
    result.forward_time = 0;
    result.backward_time = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < repeat; iter++) {
        convFwd(handle_, devPtrX, devPtrW, devPtrY, xDesc, rawWdesc, yDesc, cudnnConvDesc, alpha, beta, ySize, false);
        cudnnGetReductionWorkspaceSize(handle_, wReduceDesc, wDesc, sumWdesc, &workSpaceSize);
        if (workSpaceSize > 0) {
            checkCudaErr(cudaMalloc(&workSpace, workSpaceSize));
        }
        checkCudnnErr(cudnnReduceTensor(handle_, wReduceDesc, nullptr, 0, workSpace, workSpaceSize, 
                                        (void*)&alpha, wDesc, devPtrW, (void*)&beta, sumWdesc, devSumW));
        if (workSpaceSize > 0) {
            checkCudaErr(cudaFree(workSpace));
        }
        checkCudnnErr(cudnnPoolingForward(handle_, xPoolingDesc, 
                                          (void*)&alpha, xDesc, devPtrX, (void*)&beta, avgXdesc, devAvgX));
        checkCudaErr(cudaDeviceSynchronize());
    }
    end = std::chrono::high_resolution_clock::now();
    result.forward_start_timestamp = to_us(start.time_since_epoch());
    result.forward_time = duration_ms(start, end) / repeat;
    result.forward_end_timestamp = to_us(end.time_since_epoch());

    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < repeat; iter++) {
        checkCudnnErr(cudnnPoolingForward(handle_, gradYpoolingDesc, 
                                          (void*)&alpha, yDesc, devPtrY, (void*)&beta, avgGradYdesc, devAvgGradY));
        cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, sumWfiltDesc, avgGradYdesc, gradY2gradXConvDesc, avgXdesc, 
                                                     cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, 
                                                     &workSpaceSize);
        if (workSpaceSize > 0) {
            checkCudaErr(cudaMalloc(&workSpace, workSpaceSize));
        }
        checkCudnnErr(cudnnConvolutionBackwardData(handle_, (void*)&alpha, 
                                                   sumWfiltDesc, devSumW, 
                                                   avgGradYdesc, devAvgGradY, 
                                                   gradY2gradXConvDesc, 
                                                   cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, 
                                                   workSpace, workSpaceSize, (void*)&beta, 
                                                   avgXdesc, devAvgX));
        if (workSpaceSize > 0) {
            checkCudaErr(cudaFree(workSpace));
        }

        cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_, avgXdesc, avgGradYdesc, gradY2gradXConvDesc, sumWfiltDesc, 
                                                       cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                                                       &workSpaceSize);
        if (workSpaceSize > 0) {
            checkCudaErr(cudaMalloc(&workSpace, workSpaceSize));
        }
        cudnnConvolutionBackwardFilter(handle_, (void*)&alpha, 
                                       avgXdesc, devAvgX, 
                                       avgGradYdesc, devAvgGradY, 
                                       gradY2gradXConvDesc, 
                                       cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, 
                                       workSpace, workSpaceSize, (void*)&beta, 
                                       sumWfiltDesc, devSumW);
        if (workSpaceSize > 0) {
            checkCudaErr(cudaFree(workSpace));
        }
        checkCudaErr(cudaDeviceSynchronize());
    }
    end = std::chrono::high_resolution_clock::now();
    result.backward_start_timestamp = to_us(start.time_since_epoch());
    result.backward_time = duration_ms(start, end) / repeat;
    result.backward_end_timestamp = to_us(end.time_since_epoch());
    result.context_size = avgXsize * sizeof(devAvgX[0]);

clean:
    if (devAvgX) cudaFree(devAvgX);
    if (devAvgGradY) cudaFree(devAvgGradY);
    if (devSumW) cudaFree(devSumW);
    if (avgGradYdesc) cudnnDestroyTensorDescriptor(avgGradYdesc);
    if (avgXdesc) cudnnDestroyTensorDescriptor(avgXdesc);
    if (sumWdesc) cudnnDestroyTensorDescriptor(sumWdesc);
    if (wDesc) cudnnDestroyTensorDescriptor(wDesc);
    if (wReduceDesc) cudnnDestroyReduceTensorDescriptor(wReduceDesc);
    if (gradYpoolingDesc) cudnnDestroyPoolingDescriptor(gradYpoolingDesc);
    if (xPoolingDesc) cudnnDestroyPoolingDescriptor(xPoolingDesc);
    if (gradY2gradXConvDesc) cudnnDestroyConvolutionDescriptor(gradY2gradXConvDesc);
    return numErrors;
}

int baseConv(cudnnHandle_t handle_, 
            int xShape[], int wShape[], int yShape[], int convStride[],
            cudnnTensorDescriptor_t xDesc,
            cudnnFilterDescriptor_t wDesc,
            cudnnTensorDescriptor_t yDesc,
            float* devPtrX,
            float* devPtrY,
            float* devPtrW,
            int wStride[], int repeat, ProfileResult &result
            ) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    float forward_time = 0, backward_x_time = 0, backward_w_time = 0;
    int numErrors = 0;
    int xPad[] = {1, 1};
    int dilation[] = {1, 1};
    int strideX[4], strideW[4], strideY[4];
    int xSize, wSize, ySize;

    cudnnConvolutionDescriptor_t cudnnConvDesc;

    checkCudnnErr(cudnnCreate(&handle_));
    checkCudnnErr(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));

    xSize = xShape[0] * xShape[1] * xShape[2] * xShape[3];
    ySize = yShape[0] * yShape[1] * yShape[2] * yShape[3];
    wSize = wShape[0] * wShape[1] * wShape[2] * wShape[3];

    checkCudnnErr(cudnnSetConvolutionNdDescriptor(
        cudnnConvDesc, 2, xPad, convStride, dilation, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < repeat; iter++) {
        convFwd(handle_, 
                devPtrX, devPtrW, devPtrY, 
                xDesc, wDesc, yDesc, cudnnConvDesc, 
                0.8, 0.0, ySize, true);
        end = std::chrono::high_resolution_clock::now();
    }
    result.forward_start_timestamp = to_us(start.time_since_epoch());
    result.forward_time = duration_ms(start, end) / repeat;
    result.forward_end_timestamp = to_us(end.time_since_epoch());

    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < repeat; iter++) {
        convBwdW(handle_, 
                 devPtrX, devPtrW, devPtrY, 
                 xDesc, wDesc, yDesc, cudnnConvDesc, 
                 0.8, 0.0, wSize, false);
        convBwdX(handle_, 
                 devPtrX, devPtrW, devPtrY, 
                 xDesc, wDesc, yDesc, cudnnConvDesc, 
                 0.8, 0.0, xSize, false);
        checkCudaErr(cudaDeviceSynchronize());
    }
    end = std::chrono::high_resolution_clock::now();
    result.backward_start_timestamp = to_us(start.time_since_epoch());
    result.backward_time = duration_ms(start, end) / repeat;
    result.backward_end_timestamp = to_us(end.time_since_epoch());
    result.context_size = xSize * sizeof(devPtrX[0]);

clean:
    if (cudnnConvDesc) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
    return 0;
}

static void
initImage(int8_t* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
        image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
    }
}


int profile_conv(
    cudnnHandle_t handle_,
    int cIn, int cOut,
    int stride, int kernSize,
    int batch, int xH, int xW,
    int avgRad, int warmup, int repeat, ProfileResult &result) {
        
    int numErrors = 0;
    int xShape[4] = {batch, cIn, xH, xW};
    int wShape[4] = {cOut, cIn, kernSize, kernSize};
    int yShape[4] = {batch, cOut, xH / stride, xW / stride};
    int convStride[2] = {stride, stride};
    int strideX[4], strideW[4], strideY[4];
    int xSize, ySize, wSize;
    float* devPtrX = nullptr;
    float* devPtrY = nullptr;
    float* devPtrW = nullptr;
    float* hostPtrX = nullptr;
    float* hostPtrY = nullptr;
    float* hostPtrW = nullptr;

    cudnnTensorDescriptor_t xDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnTensorDescriptor_t yDesc;
    checkCudnnErr(cudnnCreateTensorDescriptor(&xDesc));
    checkCudnnErr(cudnnCreateFilterDescriptor(&wDesc));
    checkCudnnErr(cudnnCreateTensorDescriptor(&yDesc));
    generateStrides(xShape, strideX, 4, CUDNN_TENSOR_NCHW);
    xSize = xShape[0] * xShape[1] * xShape[2] * xShape[3];
    generateStrides(yShape, strideY, 4, CUDNN_TENSOR_NCHW);
    ySize = yShape[0] * yShape[1] * yShape[2] * yShape[3];
    generateStrides(wShape, strideW, 4, CUDNN_TENSOR_NCHW);
    wSize = wShape[0] * wShape[1] * wShape[2] * wShape[3];
    hostPtrX = (float*)calloc(xSize, sizeof(hostPtrX[0]));
    hostPtrW = (float*)calloc(wSize, sizeof(hostPtrW[0]));
    hostPtrY = (float*)calloc(ySize, sizeof(hostPtrY[0]));
    cudaMalloc((void**)&(devPtrX), xSize * sizeof(devPtrX[0]));
    cudaMalloc((void**)&(devPtrY), ySize * sizeof(devPtrY[0]));
    cudaMalloc((void**)&(devPtrW), wSize * sizeof(devPtrW[0]));
    checkCudnnErr(cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 4, xShape, strideX));
    checkCudnnErr(cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 4, yShape, strideY));
    checkCudnnErr(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, wShape));

    initImage((int8_t*)hostPtrX, xSize);
    initImage((int8_t*)hostPtrW, wSize);
    initImage((int8_t*)hostPtrY, ySize);
    checkCudaErr(cudaMemcpy(devPtrX, hostPtrX, sizeof(hostPtrX[0]) * xSize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrW, hostPtrW, sizeof(hostPtrW[0]) * wSize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrY, hostPtrY, sizeof(hostPtrY[0]) * ySize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    if (avgRad < 1) {
        numErrors += baseConv(handle_, xShape, wShape, yShape, convStride, xDesc, wDesc, yDesc, devPtrX, devPtrY, devPtrW, strideW, warmup, result);
        numErrors += baseConv(handle_, xShape, wShape, yShape, convStride, xDesc, wDesc, yDesc, devPtrX, devPtrY, devPtrW, strideW, repeat, result);
    } else {
        numErrors += avgConv(handle_, xShape, wShape, yShape, convStride, xDesc, wDesc, yDesc, devPtrX, devPtrY, devPtrW, strideW, avgRad, warmup, result);
        numErrors += avgConv(handle_, xShape, wShape, yShape, convStride, xDesc, wDesc, yDesc, devPtrX, devPtrY, devPtrW, strideW, avgRad, repeat, result);
    }
clean:
    if (devPtrX) cudaFree(devPtrX);
    if (devPtrW) cudaFree(devPtrW);
    if (devPtrY) cudaFree(devPtrY);
    if (xDesc) cudnnDestroyTensorDescriptor(xDesc);
    if (wDesc) cudnnDestroyFilterDescriptor(wDesc);
    if (yDesc) cudnnDestroyTensorDescriptor(yDesc);
    return numErrors;
}

int main(int argc, char **argv) {
    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    if (argc == 1) {
        ProfileResult baseline, ours;
        profile_conv(handle_, 512, 512, 1, 3, 32, 30, 40, 0, 5, 50, baseline);
        profile_conv(handle_, 512, 512, 1, 3, 32, 30, 40, 4, 5, 50, ours);
        printf("---------------------------------\n");
        printf("Vanilla Convolution\n");
        print_profiling_result(baseline);
        printf("---------------------------------\n");
        printf("Ours Convolution\n");
        print_profiling_result(ours);
        printf("---------------------------------\n");
        printf("Backward Speedup: %.3fx\n", baseline.backward_time / ours.backward_time);
        printf("Context Reduction: %.3fx\n", baseline.context_size * 1.0 / ours.context_size);
        printf("---------------------------------\n");
    } else if (argc == 3) {
        int repeat, warmup;
        int c_in, c_out, stride, kern_size, batch, x_h, x_w, avg_rad;
        std::ifstream cfg_file(argv[1], std::ios_base::in);
        if (!cfg_file) {
            printf("Bad config file\n");
            return -1;
        }
        std::ofstream result_file(argv[2], std::ios_base::out);
        if (!result_file) {
            printf("Cannot create output file\n");
            return -1;
        }
        cfg_file >> repeat >> warmup;
        result_file << "C_IN, C_OUT, Stride, Kernel_size, Batch, X_H, X_W, Avg_Rad, "
                    << "Forward_Time[ms], Backward_Time[ms], Total_Time[ms], "
                    << "Context_Size[KB]"
                    << "Forward_Start_Timestamp[us]" << "Forward_End_Timestamp[us]"
                    << "Backward_Start_Timestamp[us]" << "Backward_End_Timestamp[us]"
                    << std::endl;
        printf("Warmup: %d Repeat: %d\n", warmup, repeat);
        while (cfg_file >> c_in >> c_out >> stride >> kern_size >> batch >> x_h >> x_w >> avg_rad) {
            printf("Start New Test:\n");
            printf("C_In: %d C_Out: %d Stride: %d Kernel Size: %d\nBatch Size: %d X_H: %d X_W: %d Filter Size: %d\n", c_in, c_out, stride, kern_size, batch, x_h, x_w, avg_rad);
            ProfileResult result;
            profile_conv(handle_, c_in, c_out, stride, kern_size, batch, x_h, x_w, avg_rad, warmup, repeat, result);
            result_file << c_in << ", " << c_out << ", " << stride << ", " << kern_size << ", ";
            result_file << batch << ", " << x_h << ", " << x_w << ", " << avg_rad << ", ";
            result_file << result.forward_time << ", " << result.backward_time << ", ";
            result_file << result.forward_time + result.backward_time << ", ";
            result_file << result.context_size / 1024.0 << ", ";
            result_file << result.forward_start_timestamp.count() << ", " << result.forward_end_timestamp.count() << ", ";
            result_file << result.backward_start_timestamp.count() << ", " << result.backward_end_timestamp.count();
            result_file << std::endl;
            printf("Done Sleep 500 ms\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    } else {
        printf("Bad args\n");
        return -1;
    }
    return 0;
}
