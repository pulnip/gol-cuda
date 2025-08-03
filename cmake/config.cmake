if(WIN32)
    set(DEFAULT_RENDER_BACKEND "DirectX")
elseif(APPLE)
    set(DEFAULT_RENDER_BACKEND "Metal")
else()
    set(DEFAULT_RENDER_BACKEND "OpenGL")
endif()

if(NOT DEFINED RENDER_BACKEND)
    set(RENDER_BACKEND "${DEFAULT_RENDER_BACKEND}" CACHE STRING "Graphics API backend")
    set_property(CACHE RENDER_BACKEND PROPERTY STRINGS "DirectX" "Metal" "OpenGL")
endif()

if(RENDER_BACKEND STREQUAL "DirectX")
    if(APPLE)
        message(FATAL_ERROR
            "DirectX is not supported on macOS."
            "Please use other backend."
        )
    endif()
    find_program(SHADER_COMPILER NAMES fxc HINTS
        "$ENV{VCToolsInstallDir}/bin/Hostx64/x64"
        "$ENV{WindowsSdkBinPath}/x64"
        "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/x64"
        "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin"
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64"
    )
    if(NOT SHADER_COMPILER)
        message(FATAL_ERROR
            "fxc.exe (HLSL Shader Compiler) not found!"
            "Please check your Windows SDK installation."
        )
    endif()
    add_definitions(-DUSE_DIRECTX)
    # DirectX related packages
    find_package(directxmath REQUIRED CONFIG)
    # find_package(directxtk REQUIRED CONFIG)
    set(BACKEND_SOURCE_DIR "backends/dx11")
elseif(RENDER_BACKEND STREQUAL "Metal")
    if(WIN32)
        message(FATAL_ERROR
            "Metal is not supported on Windows."
            "Please use other backend."
        )
    endif()
    add_definitions(-DUSE_METAL)
    find_library(METAL_FRAMEWORK Metal)
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE MACOSX_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_OSX_SYSROOT "${MACOSX_SDK_PATH}")
    set(BACKEND_SOURCE_DIR "backends/metal")
elseif(RENDER_BACKEND STREQUAL "OpenGL")
    set(SHADER_COMPILER glslangValidator)
    add_definitions(-DUSE_OPENGL)
    # OpenGL related packages
    find_package(OpenGL REQUIRED)
    find_package(glad REQUIRED)
    set(BACKEND_SOURCE_DIR "backends/opengl")
else()
    message(FATAL_ERROR
        "Invalid RENDER_BACKEND: ${RENDER_BACKEND}"
    )
endif()
