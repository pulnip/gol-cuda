#pragma once
#include <source_location>

void checkCuda(cudaError_t error, std::source_location sl = std::source_location::current());