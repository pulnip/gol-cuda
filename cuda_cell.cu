#include <stdio.h>
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cuda_entry.h"
#include "cuda_util.cuh"

constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;
constexpr int SIZE = WIDTH * HEIGHT;

static unsigned char* dev_prev = nullptr;
static unsigned char* dev_curr = nullptr;

static cudaGraphicsResource* cudaTexResource = nullptr;

void initCellGrid(int width, int height) {
    if (dev_prev || dev_curr) return;

    checkCuda(cudaMalloc(&dev_prev, SIZE));
    checkCuda(cudaMalloc(&dev_curr, SIZE));

    unsigned char* h_init = new unsigned char[SIZE];
    for (int i = 0; i < SIZE; ++i)
        h_init[i] = (rand() & 1) ? 255 : 0; // use 0 or 255 so GL_R8 shows visible white/black

    checkCuda(cudaMemcpy(dev_prev, h_init, SIZE, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_curr, h_init, SIZE, cudaMemcpyHostToDevice));
    delete[] h_init;
}

void destroyCuda() {
    checkCuda(cudaFree(dev_prev));
    checkCuda(cudaFree(dev_curr));
    dev_prev = nullptr;
    dev_curr = nullptr;

    checkCuda(cudaDeviceReset());
}

void registerCudaTexture(GLuint texture){
    glBindTexture(GL_TEXTURE_2D, texture);
    glFinish();
    checkCuda(cudaGraphicsGLRegisterImage(&cudaTexResource,
        texture, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard));
}

void updateCellTexture() {
    cudaArray_t array;
    checkCuda(cudaGraphicsMapResources(1, &cudaTexResource));
    checkCuda(cudaGraphicsSubResourceGetMappedArray(&array,
        cudaTexResource, 0, 0));

    const size_t widthBytes = WIDTH * sizeof(unsigned char);
    checkCuda(cudaMemcpy2DToArray(array, 0, 0, dev_prev,
        widthBytes, widthBytes, HEIGHT, cudaMemcpyDeviceToDevice));

    checkCuda(cudaGraphicsUnmapResources(1, &cudaTexResource));
}

__global__ void updateKernel(unsigned char* current, unsigned char* next, int width, int height) {
    // Game of Life Logic...
}

void updateCell() {
    dim3 threads(16, 16);
    dim3 blocks((WIDTH + 15)/16, (HEIGHT + 15)/16);

    updateKernel<<<blocks, threads>>>(dev_prev, dev_curr, WIDTH, HEIGHT);
    std::swap(dev_prev, dev_curr);
}

unsigned char* getCurrentCellBuffer() {
    return dev_prev;
}