cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cu_hlt CXX CUDA)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11")

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
set(CUDA_HOST_COMPILER "g++")
set(CUDA_SEPARABLE_COMPILATION ON)

# Options for P100:
# -std=c++11;--gpu-architecture=compute_60;--gpu-code=sm_60;

set(CUDA_NVCC_FLAGS "-O3;-g;--use_fast_math;--expt-relaxed-constexpr;--generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=64;")

include_directories(cuda/include)
include_directories(main/include)
include_directories(stream/include)

add_subdirectory(cuda)
add_subdirectory(stream)
add_subdirectory(checker)

cuda_add_executable( cu_hlt
  main/src/main.cpp
  main/src/Timer.cpp
  main/src/Tools.cpp
  main/src/Logger.cpp
  )

target_link_libraries( cu_hlt tbb libcuSearchByTriplet libStream )

