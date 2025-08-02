#include <stdio.h>
#include <SDL3/SDL.h>
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>
#include <glad/glad.h>
#include "cuda_entry.h"

struct AppState{
    SDL_Window* window;
    SDL_GLContext context;

    unsigned int vertexArray, vertexBuffer;
    GLuint program;
};

GLuint loadShader(GLenum type, const char* path) {
    // 1. 파일 열기
    FILE* file = fopen(path, "rb");
    if (!file) {
        printf("Failed to open shader file: %s\n", path);
        return 0;
    }

    // 2. 파일 크기 알아내고 버퍼 만들기
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* source = (char*)malloc(length + 1);
    fread(source, 1, length, file);
    source[length] = '\0'; // 널문자 추가!
    fclose(file);

    // 3. OpenGL에 셰이더 만들고 소스 등록
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, (const char**)&source, NULL);
    glCompileShader(shader);
    free(source);

    // 4. 컴파일 에러 체크
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        if (logLength > 0) {
            char* log = (char*)malloc(logLength);
            glGetShaderInfoLog(shader, logLength, NULL, log);
            printf("Shader compile error in %s:\n%s\n", path, log);
            free(log);
        }
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint createProgram(const char* vertPath, const char* fragPath){
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
        if (logLength > 0){
            char* log = (char*)malloc(logLength);
            glGetProgramInfoLog(prog, logLength, NULL, log);
            printf("Program link error:\n%s\n", log);
            free(log);
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
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_WindowFlags flags = SDL_WINDOW_TRANSPARENT | SDL_WINDOW_BORDERLESS | SDL_WINDOW_OPENGL;

    auto window = SDL_CreateWindow("GoL-CUDA", 800, 600, flags);
    if(window == nullptr){
        SDL_Log("Couldn't create window: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }
    auto context = SDL_GL_CreateContext(window);

    if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
        SDL_Log("Failed to initialize GLAD");
        return SDL_APP_FAILURE;
    }
    SDL_GL_SetSwapInterval(1);

    unsigned int vertexArray, vertexBuffer;
    glGenVertexArrays(1, &vertexArray);
    glGenBuffers(1, &vertexBuffer);

    glBindVertexArray(vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    float lineVerts[] = {
        -0.5f, -0.5f,
         0.5f,  0.5f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(lineVerts), lineVerts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    auto prog = createProgram("vertex.glsl", "fragment.glsl");

    app->window = window;
    app->context = context;

    app->vertexArray = vertexArray;
    app->vertexBuffer = vertexBuffer;

    app->program = prog;

    *appState = app;

    return SDL_APP_CONTINUE;
}

static SDL_AppResult _handle_key_event([[maybe_unused]] void* ctx,
    SDL_Scancode key_code)
{
    switch(key_code){
    /* Quit. */
    case SDL_SCANCODE_ESCAPE:
    case SDL_SCANCODE_Q:
        return SDL_APP_SUCCESS;
    case SDL_SCANCODE_R:
        break;
    case SDL_SCANCODE_RIGHT:
        break;
    case SDL_SCANCODE_UP:
        break;
    case SDL_SCANCODE_LEFT:
        break;
    case SDL_SCANCODE_DOWN:
        break;
    default:
        break;
    }
    return SDL_APP_CONTINUE;
}


SDL_AppResult SDL_AppEvent([[maybe_unused]] void* appState,
    SDL_Event* event)
{
    switch(event->type){
    case SDL_EVENT_QUIT:
        SDL_Log("SDL_EVENT_QUIT");
        return SDL_APP_SUCCESS;
    case SDL_EVENT_KEY_DOWN:
        return _handle_key_event(nullptr, event->key.scancode);        
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appState){
    auto app = static_cast<AppState*>(appState);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(app->program);
    glBindVertexArray(app->vertexArray);
    glDrawArrays(GL_LINES, 0, 2);

    SDL_GL_SwapWindow(app->window);
    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appState, [[maybe_unused]] SDL_AppResult result){
    auto app = static_cast<AppState*>(appState);

    SDL_DestroyWindow(app->window);
    delete app;

    destroyCuda();
}