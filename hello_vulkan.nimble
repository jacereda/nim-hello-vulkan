# Package

version       = "0.1.0"
author        = "Jorge Acereda"
description   = "Hello Vulkan in Nim"
license       = "MIT"
srcDir        = "src"
bin           = @["hello_vulkan"]


# Dependencies

requires "nim >= 1.2.6"
requires "https://github.com/jacereda/nim-cv"
requires "https://github.com/jacereda/nim-vk"
requires "glm"

before build:
  exec("glslc src/shader.vert -o src/vert.spv")
  exec("glslc src/shader.frag -o src/frag.spv")
