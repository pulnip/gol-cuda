#pragma once

#include <glad/glad.h>

extern "C"
{
    void printCudaDeviceInfo();
    void destroyCuda();

    void initCell(int width, int height);
    void updateCell();

    void registerTexture(GLuint texture);
    void updateTexture();
}
