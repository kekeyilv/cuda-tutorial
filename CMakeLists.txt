cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
project(cuda-tutorial LANGUAGES CXX CUDA)

include_directories(framework)
link_libraries(curand)
add_library(framework STATIC framework/framework.cu framework/framework.cuh)

file(GLOB sources src/*.cu)
foreach(file_path ${sources})
    get_filename_component(file_name ${file_path} NAME_WE)
    add_executable(${file_name} ${file_path})
    target_link_libraries(${file_name} framework)
endforeach()

