#include <cstdio>
#include <source_location>
#include "cuda_util.cuh"

void checkCuda(cudaError_t error, std::source_location sl){
    if(error != cudaSuccess){
        fprintf(stderr, "CUDA error at %s:%d in %s: %s\n",
            sl.file_name(), sl.line(), sl.function_name(), cudaGetErrorString(error));
        std::exit(1);
    }
}
