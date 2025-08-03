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
add_library(glad_release STATIC
    vendor/glad_release/src/glad.c
)
target_include_directories(glad_release PUBLIC
    vendor/glad_release/include
)
add_library(glad_debug STATIC
    vendor/glad_debug/src/glad.c
)
target_include_directories(glad_debug PUBLIC
    vendor/glad_debug/include
)

add_library(${PROJECT_NAME}_KERNEL
    cuda_util.cu
    cuda_check.cu
    cuda_cell.cu
)
target_include_directories(${PROJECT_NAME}_KERNEL PRIVATE
    $<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/vendor/glad_debug/include>
    $<$<NOT:$<CONFIG:Debug>>:${CMAKE_SOURCE_DIR}/vendor/glad_release/include>
)

add_executable(${PROJECT_NAME}
    main.cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE
    "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}_KERNEL
    SDL3::SDL3
    $<$<CONFIG:Debug>:glad_debug>
    $<$<NOT:$<CONFIG:Debug>>:glad_release>
)
