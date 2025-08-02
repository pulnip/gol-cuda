#ifndef MY_CUDA_ENTRY_H
#define MY_CUDA_ENTRY_H

extern "C"
{
    void printCudaDeviceInfo();
    void destroyCuda();
}

#endif // MY_CUDA_ENTRY_H