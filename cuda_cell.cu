#include <stdio.h>
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cuda_entry.h"
#include "cuda_util.cuh"
#include <curand_kernel.h>

int gWidth, gHeight;

static unsigned char* dev_prev = nullptr;
static unsigned char* dev_curr = nullptr;

static cudaGraphicsResource* cudaTexResource = nullptr;

void destroyCuda(){
    checkCuda(cudaFree(dev_prev));
    checkCuda(cudaFree(dev_curr));
    dev_prev = nullptr;
    dev_curr = nullptr;

    checkCuda(cudaDeviceReset());
}

__global__
void initKernel(unsigned char* grid, int width, int height){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x >= width || y >= height)
        return;

    int idx = y * width + x;
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    grid[idx] = (curand(&state) & 1) ? 255 : 0;
}

__global__
void updateKernel(unsigned char* current, unsigned char* next, int width, int height){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x >= width || y >= height)
        return;

    int count = 0;
    for(int dy = -1; dy <= 1; ++dy){
        for(int dx = -1; dx <= 1; ++dx){
            if(dx == 0 && dy == 0)
                continue;
            int nx = (x + dx) % width;
            int ny = (y + dy) % height;

            int idx = ny*width + nx;
            // Alive if positive
            count += (current[idx] > 0);
        }
    }

    // Rule of Game of Life
    int idx = y*width + x;
    if(current[idx]){
        // if Alive
        next[idx] = (count==2 || count==3) ? 255 : 0;
    } else {
        // if Dead
        next[idx] = (count==3) ? 255 : 0;
    }
}

void initCell(int width, int height){
    if(dev_prev || dev_curr)
        return;

    gWidth = width;
    gHeight = height;

    int size = width * height;

    checkCuda(cudaMalloc(&dev_prev, size));
    checkCuda(cudaMalloc(&dev_curr, size));

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    initKernel<<<blocks, threads>>>(dev_prev, width, height);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(dev_curr, dev_prev, size, cudaMemcpyDeviceToDevice));
}

void updateCell(){
    dim3 threads(16, 16);
    dim3 blocks((gWidth + 15)/16, (gHeight + 15)/16);

    updateKernel<<<blocks, threads>>>(dev_prev, dev_curr, gWidth, gHeight);
    std::swap(dev_prev, dev_curr);
}

void registerTexture(GLuint texture){
    glBindTexture(GL_TEXTURE_2D, texture);
    glFinish();
    checkCuda(cudaGraphicsGLRegisterImage(&cudaTexResource,
        texture, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard));
}

void updateTexture(){
    cudaArray_t array;
    checkCuda(cudaGraphicsMapResources(1, &cudaTexResource));
    checkCuda(cudaGraphicsSubResourceGetMappedArray(&array,
        cudaTexResource, 0, 0));

    const size_t widthBytes = gWidth * sizeof(unsigned char);
    checkCuda(cudaMemcpy2DToArray(array, 0, 0, dev_prev,
        widthBytes, widthBytes, gHeight, cudaMemcpyDeviceToDevice));

    checkCuda(cudaGraphicsUnmapResources(1, &cudaTexResource));
}
