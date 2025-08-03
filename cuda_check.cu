#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_util.cuh"

extern "C"
{

void printCudaDeviceInfo(){
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess){
        fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n",
            static_cast<int>(error_id), cudaGetErrorString(error_id));
        return;
    }

    if (deviceCount == 0)
        printf("No CUDA devices found.\n");
    else
        printf("Found %d CUDA devices.\n", deviceCount);

    int driverVersion = 0;
    int runtimeVersion = 0;
    checkCuda(cudaDriverGetVersion(&driverVersion));
    checkCuda(cudaRuntimeGetVersion(&runtimeVersion));

    for (int device=0; device<deviceCount; ++device){
        cudaDeviceProp deviceProp;
        checkCuda(cudaGetDeviceProperties(&deviceProp, device));

        printf("Device %d - %s\n", device, deviceProp.name);
        printf("  CUDA Driver Version / Runtime Version:         %d.%d / %d.%d\n",
            driverVersion / 1000, (driverVersion % 1000) / 10,
            runtimeVersion / 1000, (runtimeVersion % 1000));
        printf("  Total Global Memory:                           %zu MB\n",
            deviceProp.totalGlobalMem / (1024 * 1024));
        printf("  GPU Clock Rate:                                %llf MHz\n",
            deviceProp.clockRate * 1e-3f);
        printf("  Memory Clock Rate:                             %llf MHz\n",
            deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d bits\n",
            deviceProp.memoryBusWidth);
        printf("  L2 Cache Size:                                 %d KB\n",
            deviceProp.l2CacheSize / 1024);
        printf("  Max Texture Dimension Size (x,y,z):            %d, %d, %d\n",
            deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture3D[0]);
        printf("  Max Layered Texture Size (dim) x layers:       %d x %d\n",
            deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1]);
        printf("  Total Constant Memory:                         %zu KB\n",
            deviceProp.totalConstMem / 1024);
        printf("  Unified Addressing:                            %s\n",
            (deviceProp.unifiedAddressing ? "Yes" : "No"));
        printf("  Concurrent Kernels:                            %s\n",
            (deviceProp.concurrentKernels ? "Yes" : "No"));
        printf("  ECC Enabled:                                   %s\n",
            (deviceProp.ECCEnabled ? "Yes": "No"));
        printf("  Compute Capability:                            %d.%d\n",
            deviceProp.major, deviceProp.minor);

        printf("  Async Engine Count:                            %d\n",
            deviceProp.asyncEngineCount);

        printf("  Number of Multiprocessors:                     %d\n",
            deviceProp.multiProcessorCount);
        printf("  Maximum Threads per MultiProcessor:            %d\n",
            deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum Threads per Block:                     %d\n",
            deviceProp.maxThreadsPerBlock);
        printf("  Maximum Blocks Per Multiprocessor:             %d\n",
            deviceProp.maxBlocksPerMultiProcessor);
        printf("  Maximum Threads per Dimension (x, y, z):       %d, %d, %d\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Warp size in threads:                          %d\n", deviceProp.warpSize);
        printf("  Maximum Grid Size (x,y,z):                     %d, %d, %d\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }
}

}