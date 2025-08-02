#version 460 core

in vec2 vUV;
out vec4 FragColor;

uniform sampler2D tex;

void main() {
    float value = texture(tex, vUV).r;
    FragColor = vec4(value, value, value, 1.0);
}