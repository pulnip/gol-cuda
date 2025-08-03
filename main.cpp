#include <stdio.h>
#include <fstream>
#include <print>
#include <sstream>
#include <SDL3/SDL.h>
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>
#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_opengl3.h>
#include "cuda_entry.h"

constexpr int WINDOW_WIDTH = 800, WINDOW_HEIGHT = 600;
constexpr auto CELL_WIDTH = 800, CELL_HEIGHT = 600;

struct AppState{
    SDL_Window* window;
    SDL_GLContext context;

    unsigned int vertexArray, vertexBuffer;
    GLuint program;
    GLuint texture;
};

GLuint loadShader(GLenum type, const std::string& path){
    std::ifstream file(path, std::ios::in);
    if (!file.is_open()){
        std::println("Failed to open shader file: {}", path);
        return 0;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    std::string source = buffer.str();
    const char* sourcePtr = source.c_str();

    // compile shader
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &sourcePtr, NULL);
    glCompileShader(shader);

    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if(!compiled){
        GLint logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        if(logLength > 0){
            std::string log(logLength, '\0');
            glGetShaderInfoLog(shader, logLength, NULL, log.data());
            std::println("Shader compile error in {}: {}", path, log);
        }
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint createProgram(const std::string& vertPath, const std::string& fragPath){
    GLuint vert = loadShader(GL_VERTEX_SHADER, vertPath);
    GLuint frag = loadShader(GL_FRAGMENT_SHADER, fragPath);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if(!linked){
        GLint logLength = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLength);
        if(logLength > 0){
            std::string log(logLength, '\0');
            glGetProgramInfoLog(prog, logLength, NULL, log.data());
            std::println("Program link error: {}", log);
        }
        glDeleteProgram(prog);
        return 0;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);

    return prog;
}

SDL_AppResult SDL_AppInit([[maybe_unused]] void** appState,
    [[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    printCudaDeviceInfo();

    if(!SDL_SetAppMetadata("GoL-CUDA", "0.1", "com.example.golcuda"))
        return SDL_APP_FAILURE;
    if(!SDL_Init(SDL_INIT_VIDEO)){
        SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    AppState* app = new AppState();
    if(app==nullptr){
        SDL_Log("Couldn't create AppState");
    }

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
        SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_WindowFlags flags = SDL_WINDOW_OPENGL |
        SDL_WINDOW_TRANSPARENT | SDL_WINDOW_BORDERLESS;

    auto window = SDL_CreateWindow("GoL-CUDA",
        WINDOW_WIDTH, WINDOW_HEIGHT, flags);
    if(window == nullptr){
        SDL_Log("Couldn't create window: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }
    auto context = SDL_GL_CreateContext(window);

    if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
        SDL_Log("Failed to initialize GLAD");
        return SDL_APP_FAILURE;
    }
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    SDL_GL_SetSwapInterval(1);

    unsigned int vertexArray, vertexBuffer;
    glGenVertexArrays(1, &vertexArray);
    glGenBuffers(1, &vertexBuffer);

    glBindVertexArray(vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    float quadVerts[] = {
        //  Pos      | UV
       -1.0f, -1.0f,  0.0f, 0.0f,   // Bottom-Left
        1.0f, -1.0f,  1.0f, 0.0f,   // Bottom-Right
       -1.0f,  1.0f,  0.0f, 1.0f,   //    Top-Left
        1.0f,  1.0f,  1.0f, 1.0f    //    Top-Right
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts),
        quadVerts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
        4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
        4*sizeof(float), (void*)(2*sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    auto prog = createProgram("vertex.glsl", "fragment.glsl");

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glFinish();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, CELL_WIDTH, CELL_HEIGHT, 0,
        GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    registerTexture(texture);
    initCell(CELL_WIDTH, CELL_HEIGHT);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplSDL3_InitForOpenGL(window, context);
    ImGui_ImplOpenGL3_Init();

    app->window = window;
    app->context = context;

    app->vertexArray = vertexArray;
    app->vertexBuffer = vertexBuffer;

    app->program = prog;
    app->texture = texture;

    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);

    *appState = app;

    return SDL_APP_CONTINUE;
}

static SDL_AppResult _handle_key_event([[maybe_unused]] void* appState,
    SDL_Scancode key_code)
{
    switch(key_code){
    /* Quit. */
    case SDL_SCANCODE_ESCAPE:
        return SDL_APP_SUCCESS;
    default:
        break;
    }
    return SDL_APP_CONTINUE;
}


SDL_AppResult SDL_AppEvent(void* appState, SDL_Event* event){
    ImGui_ImplSDL3_ProcessEvent(event);
    switch(event->type){
    case SDL_EVENT_QUIT:
        SDL_Log("SDL_EVENT_QUIT");
        return SDL_APP_SUCCESS;
    case SDL_EVENT_KEY_DOWN:
        return _handle_key_event(appState, event->key.scancode);        
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appState){
    auto app = static_cast<AppState*>(appState);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    static bool paused = false;
    if(ImGui::Button(paused ? "Play" : "pause")){
        paused = !paused;
        syncCell();
        updateTexture();
    }
    ImGui::SameLine();
    if(ImGui::Button("Init")){
        initCell(CELL_WIDTH, CELL_HEIGHT);
        updateTexture();
    }

    static int updatePerSecond = 60;
    static uint64_t lastUpdate = SDL_GetTicks();
    ImGui::SliderInt("fps", &updatePerSecond, 1, 60);
    uint64_t delayMs = 1'000 / updatePerSecond;

    if(!paused && SDL_GetTicks()-lastUpdate > delayMs){
        updateTexture();
        updateCell();
        lastUpdate = SDL_GetTicks();
    }

    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(app->program);
    glBindVertexArray(app->vertexArray);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, app->texture);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    SDL_GL_SwapWindow(app->window);
    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appState, [[maybe_unused]] SDL_AppResult result){
    auto app = static_cast<AppState*>(appState);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    SDL_DestroyWindow(app->window);
    delete app;

    destroyCuda();
}