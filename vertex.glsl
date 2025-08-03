#version 460 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;

out vec2 vUV;

uniform float uZoom;

void main() {
    vec2 zoomedPos = aPos * uZoom;
    gl_Position = vec4(zoomedPos, 0.0, 1.0);
    vUV = aUV;
}