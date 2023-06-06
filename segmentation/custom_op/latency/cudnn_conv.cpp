#include <cassert>
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

static void
initImage(int8_t* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
        image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
    }
}

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

enum cudnnTransformNCHWtype { CUDNN_NO_TRANSFORM, CUDNN_TRANSFORM_FROM_NCHW, CUDNN_TRANSFORM_TO_NCHW };

int convForward(cudnnHandle_t handle_,
                float* devPtrX,
                float* devPtrW,
                float* devPtrY,
                float* hostPtrX,
                float* hostPtrW,
                float* hostPtrY,
                cudnnTensorDescriptor_t cudnnXdesc,
                cudnnFilterDescriptor_t cudnnWdesc,
                cudnnTensorDescriptor_t cudnnYdesc,
                cudnnConvolutionDescriptor_t cudnnConvDesc,
                float alpha, float beta, int ySize
) {
    void* workSpace = 0;
    size_t workSpaceSize;
    int numErrors = 0;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
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
    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(hostPtrY, devPtrY, sizeof(hostPtrY[0]) * ySize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
clean:
    if (workSpace) cudaFree(workSpace);
    return numErrors;
}

int convBackwardW(cudnnHandle_t handle_,
                  float* devPtrX,
                  float* devPtrW,
                  float* devPtrY,
                  float* hostPtrX,
                  float* hostPtrW,
                  float* hostPtrY,
                  cudnnTensorDescriptor_t cudnnXdesc,
                  cudnnFilterDescriptor_t cudnnWdesc,
                  cudnnTensorDescriptor_t cudnnYdesc,
                  cudnnConvolutionDescriptor_t cudnnConvDesc,
                  float alpha, float beta, int wSize
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
    checkCudaErr(cudaDeviceSynchronize());
    // checkCudaErr(cudaMemcpy(hostPtrW, devPtrW, sizeof(hostPtrW[0]) * wSize, cudaMemcpyDeviceToHost));
    // checkCudaErr(cudaDeviceSynchronize());
    
clean:
    if (workSpace) cudaFree(workSpace);
    return numErrors;
}

int convBackwardX(cudnnHandle_t handle_,
                  float* devPtrX,
                  float* devPtrW,
                  float* devPtrY,
                  float* hostPtrX,
                  float* hostPtrW,
                  float* hostPtrY,
                  cudnnTensorDescriptor_t cudnnXdesc,
                  cudnnFilterDescriptor_t cudnnWdesc,
                  cudnnTensorDescriptor_t cudnnYdesc,
                  cudnnConvolutionDescriptor_t cudnnConvDesc,
                  float alpha, float beta, int xSize
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

    checkCudaErr(cudaDeviceSynchronize());
clean:
    if (workSpace) cudaFree(workSpace);
    return numErrors;

}

int initMemory(float* devPtrX,
               float* devPtrW,
               float* devPtrY,
               float* hostPtrX,
               float* hostPtrW,
               float* hostPtrY,
               int xSize, int wSize, int ySize) {
    int numErrors = 0;
    initImage((int8_t*)hostPtrX, xSize);
    initImage((int8_t*)hostPtrW, wSize);
    initImage((int8_t*)hostPtrY, ySize);
    checkCudaErr(cudaMemcpy(devPtrX, hostPtrX, sizeof(hostPtrX[0]) * xSize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrW, hostPtrW, sizeof(hostPtrW[0]) * wSize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrY, hostPtrY, sizeof(hostPtrY[0]) * ySize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());
clean:
    return numErrors;
}

int avgConv(cudnnHandle_t handle_, 
            int xShape[], int wShape[], int yShape[], int convStride[],
            cudnnTensorDescriptor_t xDesc,
            cudnnTensorDescriptor_t yDesc,
            float* devPtrX,
            float* devPtrY,
            float* devPtrW,
            int wStride[]
            ) {
    int numErrors = 0;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    float prepareTime, gradXtime, gradWtime;
    cudnnTensorDescriptor_t wDesc;

    const int gradFilterSize[2] = {4, 4};
    int xPoolingSize[2];
    int xPad[2], gradYpad[2];
    int avgGradYshape[4], avgXshape[4], sumWshape[4];
    int tempShape[4];

    cudnnTensorDescriptor_t avgGradYdesc;
    cudnnTensorDescriptor_t avgXdesc;
    cudnnTensorDescriptor_t sumWdesc;
    cudnnFilterDescriptor_t sumWfiltDesc;
    int avgGradYstride[4], avgXstride[4], sumWstride[4];
    int avgGradYsize = 1, avgXsize = 1, sumWsize = 1;

    float* devAvgX;
    float* devAvgGradY;
    float* devSumW;
    // float* hostAvgX;
    // float* hostAvgGradY;

    cudnnReduceTensorDescriptor_t wReduceDesc;
    cudnnPoolingDescriptor_t gradYpoolingDesc;
    cudnnPoolingDescriptor_t xPoolingDesc;
    cudnnConvolutionDescriptor_t gradY2gradXConvDesc;

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
    for (int iter = 0; iter < 10; iter++) {
        start = std::chrono::high_resolution_clock::now();
        cudnnGetReductionWorkspaceSize(handle_, wReduceDesc, wDesc, sumWdesc, &workSpaceSize);
        if (workSpaceSize > 0) {
            checkCudaErr(cudaMalloc(&workSpace, workSpaceSize));
        }
        checkCudnnErr(cudnnReduceTensor(handle_, wReduceDesc, nullptr, 0, workSpace, workSpaceSize, 
                                        (void*)&alpha, wDesc, devPtrW, (void*)&beta, sumWdesc, devSumW));
        checkCudaErr(cudaFree(workSpace));
        checkCudnnErr(cudnnPoolingForward(handle_, gradYpoolingDesc, 
                                          (void*)&alpha, yDesc, devPtrY, (void*)&beta, avgGradYdesc, devAvgGradY));

        checkCudnnErr(cudnnPoolingForward(handle_, xPoolingDesc, 
                                          (void*)&alpha, xDesc, devPtrX, (void*)&beta, avgXdesc, devAvgX));
        checkCudaErr(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        prepareTime += (duration_ms(start, end) - prepareTime) / (iter + 1);

        // Todo: Change to correct gradX memory
        start = std::chrono::high_resolution_clock::now();
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
        checkCudaErr(cudaFree(workSpace));
        checkCudaErr(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        gradXtime += (duration_ms(start, end) - gradXtime) / (iter + 1);

        start = std::chrono::high_resolution_clock::now();
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
        checkCudaErr(cudaFree(workSpace));
        checkCudaErr(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        gradWtime += (duration_ms(start, end) - gradWtime) / (iter + 1);
    }
    printf("Ours Conv Backward Prepare: %.3f ms\n", prepareTime);
    printf("Ours Conv Backward W: %.3f ms\n", gradWtime);
    printf("Ours Conv Backward X: %.3f ms\n", gradXtime);

clean:
    if (devAvgX) cudaFree(devAvgX);
    if (devAvgGradY) cudaFree(devAvgGradY);
    if (devSumW) cudaFree(devSumW);
    // if (hostAvgX) free(hostAvgX);
    // if (hostAvgGradY) free(hostAvgGradY);
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


int main(int argc, char** argv) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    float forward_time = 0, backward_x_time = 0, backward_w_time = 0;
    int numErrors = 0;
    int xShape[] = {32, 512, 30, 40};
    int wShape[] = {512, 512, 3, 3};
    int xPad[] = {1, 1};
    int convStride[] = {1, 1};
    int dilation[] = {1, 1};
    int strideX[4], strideW[4], strideY[4];
    int xSize, wSize, ySize;
    float* devPtrX = nullptr;
    float* devPtrY = nullptr;
    float* devPtrW = nullptr;
    float* devPtrReorderedW = nullptr;
    float* hostPtrX = nullptr;
    float* hostPtrW = nullptr;
    float* hostPtrY = nullptr;

    cudnnTensorFormat_t wFormat = CUDNN_TENSOR_NCHW;
    bool fold = false;
    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;
    cudnnTransformNCHWtype transformNCHWType = CUDNN_NO_TRANSFORM;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    int device, ret = 0;
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    int deviceVer = devProp.major * 10 + devProp.minor;

    // conv fwd
    int yShape[4];
    yShape[0] = xShape[0];
    yShape[1] = wShape[0];
    for (int dim = 2; dim < 4; dim++) {
        yShape[dim] = (xShape[dim] + 2 * xPad[dim - 2]) - wShape[dim] / convStride[dim - 2] + 1;
    }

    cudnnHandle_t handle_;
    cudnnTensorDescriptor_t cudnnXdesc;
    cudnnFilterDescriptor_t cudnnWdesc;
    cudnnTensorDescriptor_t cudnnYdesc;
    cudnnConvolutionDescriptor_t cudnnConvDesc;

    checkCudnnErr(cudnnCreate(&handle_));
    checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnXdesc));
    checkCudnnErr(cudnnCreateFilterDescriptor(&cudnnWdesc));
    checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnYdesc));
    checkCudnnErr(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));


    generateStrides(xShape, strideX, 4, wFormat);
    xSize = xShape[0] * xShape[1] * xShape[2] * xShape[3];
    generateStrides(yShape, strideY, 4, wFormat);
    ySize = yShape[0] * yShape[1] * yShape[2] * yShape[3];
    generateStrides(wShape, strideW, 4, wFormat);
    wSize = wShape[0] * wShape[1] * wShape[2] * wShape[3];

    start = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)&(devPtrX), xSize * sizeof(devPtrX[0]));
    cudaMalloc((void**)&(devPtrY), ySize * sizeof(devPtrY[0]));
    cudaMalloc((void**)&(devPtrW), wSize * sizeof(devPtrW[0]));
    cudaMalloc((void**)&(devPtrReorderedW), wSize * sizeof(devPtrW[0]));
    hostPtrX = (float*)calloc(xSize, sizeof(hostPtrX[0]));
    hostPtrW = (float*)calloc(wSize, sizeof(hostPtrW[0]));
    hostPtrY = (float*)calloc(ySize, sizeof(hostPtrY[0]));
    end = std::chrono::high_resolution_clock::now();
    printf("Operator Create Time: %.3f ms\n", duration_ms(start, end));

    initImage((int8_t*)hostPtrX, xSize);
    initImage((int8_t*)hostPtrW, wSize);
    initImage((int8_t*)hostPtrY, ySize);

    start = std::chrono::high_resolution_clock::now();
    initMemory(devPtrX, devPtrW, devPtrY, hostPtrX, hostPtrW, hostPtrY, xSize, wSize, ySize);
    end = std::chrono::high_resolution_clock::now();
    printf("CPU->GPU memcpy Time: %.3f ms\n", duration_ms(start, end));

    checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnXdesc, dataType, 4, xShape, strideX));
    checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnYdesc, dataType, 4, yShape, strideY));
    checkCudnnErr(cudnnSetFilterNdDescriptor(cudnnWdesc, dataType, wFormat, 4, wShape));

    checkCudnnErr(cudnnSetConvolutionNdDescriptor(
        cudnnConvDesc, 2, xPad, convStride, dilation, mode, CUDNN_DATA_FLOAT));

    for (int iter = 0; iter < 10; iter++) {
        initMemory(devPtrX, devPtrW, devPtrY, hostPtrX, hostPtrW, hostPtrY, xSize, wSize, ySize);
        checkCudaErr(cudaDeviceSynchronize());
        start = std::chrono::high_resolution_clock::now();
        convForward(handle_, 
                    devPtrX, devPtrW, devPtrY, 
                    hostPtrX, hostPtrW, hostPtrY, 
                    cudnnXdesc, cudnnWdesc, cudnnYdesc, cudnnConvDesc, 
                    0.8, 0.0, ySize);
        end = std::chrono::high_resolution_clock::now();
        // printf("Conv Forward Total: %.3f ms\n", duration_ms(start, end));
        forward_time += (duration_ms(start, end) - forward_time) / (iter + 1);

        initMemory(devPtrX, devPtrW, devPtrY, hostPtrX, hostPtrW, hostPtrY, xSize, wSize, ySize);
        checkCudaErr(cudaDeviceSynchronize());
        start = std::chrono::high_resolution_clock::now();
        convBackwardW(handle_, 
                      devPtrX, devPtrW, devPtrY, 
                      hostPtrX, hostPtrW, hostPtrY, 
                      cudnnXdesc, cudnnWdesc, cudnnYdesc, cudnnConvDesc, 
                      0.8, 0.0, wSize);
        end = std::chrono::high_resolution_clock::now();
        // printf("Conv Backward W Total: %.3f ms\n", duration_ms(start, end));
        backward_w_time += (duration_ms(start, end) - backward_w_time) / (iter + 1);

        initMemory(devPtrX, devPtrW, devPtrY, hostPtrX, hostPtrW, hostPtrY, xSize, wSize, ySize);
        checkCudaErr(cudaDeviceSynchronize());
        start = std::chrono::high_resolution_clock::now();
        convBackwardX(handle_, 
                      devPtrX, devPtrW, devPtrY, 
                      hostPtrX, hostPtrW, hostPtrY, 
                      cudnnXdesc, cudnnWdesc, cudnnYdesc, cudnnConvDesc, 
                      0.8, 0.0, xSize);
        end = std::chrono::high_resolution_clock::now();
        // printf("Conv Backward X Total: %.3f ms\n", duration_ms(start, end));
        backward_x_time += (duration_ms(start, end) - backward_x_time) / (iter + 1);
    }
    printf("CUDNN Baseline:\n");
    printf("Conv Forward: %.3f ms\nConv Backward W: %.3f ms\nConv Backward X: %.3f ms\n", forward_time, backward_w_time, backward_x_time);

    printf("Run ours bwd\n");
    avgConv(handle_, xShape, wShape, yShape, convStride, cudnnXdesc, cudnnYdesc, devPtrX, devPtrY, devPtrW, strideW);


clean:
    if (devPtrX) cudaFree(devPtrX);
    if (devPtrReorderedW) cudaFree(devPtrReorderedW);
    if (devPtrW) cudaFree(devPtrW);
    if (devPtrY) cudaFree(devPtrY);
    if (hostPtrX) free(hostPtrX);
    if (hostPtrW) free(hostPtrW);
    if (hostPtrY) free(hostPtrY);
    if (cudnnXdesc) cudnnDestroyTensorDescriptor(cudnnXdesc);
    if (cudnnWdesc) cudnnDestroyFilterDescriptor(cudnnWdesc);
    if (cudnnYdesc) cudnnDestroyTensorDescriptor(cudnnYdesc);
    if (cudnnConvDesc) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
    if (handle_) cudnnDestroy(handle_);
    return 0;
}