#version 450
#extension GL_ARB_separate_shader_objects : enable
layout(binding = 0) uniform UniformBufferObject {
  mat4 mvp;
} ubo;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aCol;
layout(location = 0) out vec4 color;

void main() {
	gl_Position = ubo.mvp * vec4(aPos, 1.0);
	color = aCol;
}
