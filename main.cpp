#include <SDL3/SDL.h>
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>
#include "cuda_entry.h"

struct AppState{
    SDL_Window* window;
};

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

    SDL_WindowFlags flags = /* SDL_WINDOW_TRANSPARENT | SDL_WINDOW_BORDERLESS | */ SDL_WINDOW_OPENGL;

    auto window = SDL_CreateWindow("GoL-CUDA", 800, 600, flags);
    if(window == nullptr){
        SDL_Log("Couldn't create window: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    app->window = window;
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
    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appState, [[maybe_unused]] SDL_AppResult result){
    AppState* app = static_cast<AppState*>(appState);

    SDL_DestroyWindow(app->window);
    delete app;

    destroyCuda();
}