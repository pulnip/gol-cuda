# Library options
option(ImGui_DEMO
    "Include the ImGui demo window implementation in library"
    ON
)

find_path(ImGui_DIR
    NAMES imgui.h
    PATHS ${CMAKE_SOURCE_DIR}/vendor/imgui
)
if(NOT ImGui_DIR)
    message(FATAL_ERROR "ImGui header Not Found!")
else()
    set(ImGui_FOUND TRUE)
    message(STATUS "ImGui found. Headers: ${ImGui_DIR}")
endif()

add_library(ImGui STATIC
    ${ImGui_DIR}/imgui.cpp
    ${ImGui_DIR}/imgui_draw.cpp
    ${ImGui_DIR}/imgui_tables.cpp
    ${ImGui_DIR}/imgui_widgets.cpp
    $<$<BOOL:ImGui_DEMO>:${ImGui_DIR}/imgui_demo.cpp>
    ${ImGui_DIR}/backends/imgui_impl_sdl3.cpp
)
target_compile_definitions(ImGui PRIVATE
    $<IF:$<CONFIG:DEBUG>, _DEBUG, NDEBUG>
)
target_include_directories(ImGui PUBLIC
    ${ImGui_DIR}
    ${ImGui_DIR}/backends
)
target_include_directories(ImGui PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor/SDL/include
)
target_link_libraries(ImGui PRIVATE
    SDL3::SDL3
)

if(ImGui_RENDER_BACKEND STREQUAL "DirectX")
    target_sources(ImGui PRIVATE
        ${ImGui_DIR}/backends/imgui_impl_dx11.cpp
    )
elseif(ImGui_RENDER_BACKEND STREQUAL "Metal")
    target_sources(ImGui PRIVATE
        ${ImGui_DIR}/backends/imgui_impl_metal.mm
    )
    target_compile_definitions(ImGui PRIVATE
        IMGUI_IMPL_METAL_CPP
    )
    target_include_directories(ImGui PRIVATE
        ${CMAKE_SOURCE_DIR}/vendor/metal-cpp
        /opt/homebrew/include
    )
elseif(ImGui_RENDER_BACKEND STREQUAL "OpenGL")
    target_sources(ImGui PRIVATE
        ${ImGui_DIR}/backends/imgui_impl_opengl3.cpp
    )
else()
    message(FATAL_ERROR
        "Invalid ImGui_RENDER_BACKEND: ${ImGui_RENDER_BACKEND}"
    )
endif()
