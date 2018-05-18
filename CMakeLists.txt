cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cu_hlt CXX)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11 -march=native")

find_package(CUDA REQUIRED)
set(CUDA_HOST_COMPILER "g++")
set(CUDA_SEPARABLE_COMPILATION ON)
option(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Options for P100:
# --gpu-architecture=compute_60;--gpu-code=sm_60;

set(CUDA_NVCC_FLAGS "-O3;-g;-G;--use_fast_math;--expt-relaxed-constexpr;--generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=64;")

add_subdirectory(cuda)
add_subdirectory(stream)
add_subdirectory(x86/velo/clustering)
add_subdirectory(checker)

# Files from source directories
file(GLOB main_sources "main/src/*cpp")

cuda_add_executable(cu_hlt ${main_sources})

target_link_libraries(cu_hlt tbb libcuVelo libStream libx86Clustering libTrackChecking)

# option to do MC truth matching
if ( MC_CHECK )
  add_definitions( -DMC_CHECK )
endif()
