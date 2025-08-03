add_library(glad STATIC
    $<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/vendor/glad_debug/src/glad.c>
    $<$<NOT:$<CONFIG:Debug>>:${CMAKE_SOURCE_DIR}/vendor/glad_release/src/glad.c>

)
target_include_directories(glad PUBLIC
    $<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/vendor/glad_debug/include>
    $<$<NOT:$<CONFIG:Debug>>:${CMAKE_SOURCE_DIR}/vendor/glad_release/include>
)
