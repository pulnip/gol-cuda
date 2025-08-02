cmake_minimum_required(VERSION 3.28)

project(GoL_CUDA LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \"/utf-8\"")

add_subdirectory(vendor/SDL EXCLUDE_FROM_ALL)
add_library(glad STATIC vendor/glad/src/glad.c)
target_include_directories(glad PUBLIC vendor/glad/include)

add_executable(${PROJECT_NAME}
    cuda_check.cu
    main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    vendor/SDL/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    SDL3::SDL3
    glad
)
