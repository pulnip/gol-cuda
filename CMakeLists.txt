cmake_minimum_required(VERSION 3.28)

project(GoL_CUDA LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \"/utf-8\"")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(vendor/SDL EXCLUDE_FROM_ALL)
add_library(glad STATIC vendor/glad/src/glad.c)
target_include_directories(glad PUBLIC vendor/glad/include)

add_library(${PROJECT_NAME}_KERNEL
    cuda_check.cu
)

add_executable(${PROJECT_NAME}
    main.cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE
    "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)
target_include_directories(${PROJECT_NAME} PRIVATE
    vendor/SDL/include
    vendor/glad/include
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}_KERNEL
    SDL3::SDL3
    glad
)
