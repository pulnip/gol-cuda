#pragma once

#include <glad/glad.h>

extern "C"
{
    void printCudaDeviceInfo();
    void destroyCuda();

    void initCellGrid(int width, int height);
    void updateCell();
    unsigned char* getCurrentCellBuffer();

    void registerCudaTexture(GLuint texture);
    void updateCellTexture();
}
