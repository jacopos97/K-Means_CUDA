cmake_minimum_required(VERSION 3.26)
project(K-Means_CUDA LANGUAGES CUDA CXX C)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(k-means_cuda k-means_cuda.cu)

set(libraries ${CMAKE_CURRENT_SOURCE_DIR}/libraries)

add_library(ini ${libraries}/ini.c)
add_library(INIReader ${libraries}/INIReader.cpp)

target_include_directories(ini PUBLIC ${libraries})
target_include_directories(INIReader PUBLIC ${libraries})

target_link_libraries(k-means_cuda PRIVATE ini)
target_link_libraries(k-means_cuda PRIVATE INIReader)