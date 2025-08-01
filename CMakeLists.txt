cmake_minimum_required(VERSION 3.28)

project(GoF_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \"/utf-8\"")

add_executable(${PROJECT_NAME}
    cuda_check.cu
)
# add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

