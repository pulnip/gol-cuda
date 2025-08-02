cmake_minimum_required(VERSION 3.28)

project(GoL_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \"/utf-8\"")

add_subdirectory(vendor/SDL EXCLUDE_FROM_ALL)

add_executable(${PROJECT_NAME}
    cuda_check.cu
    main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    vendor/SDL/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    SDL3::SDL3
)
