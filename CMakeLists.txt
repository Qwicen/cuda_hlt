cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cu_hlt CXX CUDA)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11 -march=native")

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
set(CUDA_HOST_COMPILER "g++")
set(CUDA_SEPARABLE_COMPILATION ON)
option(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Options for P100:
# -std=c++11;--gpu-architecture=compute_60;--gpu-code=sm_60;

set(CUDA_NVCC_FLAGS "-O3;-g;--use_fast_math;--expt-relaxed-constexpr;--generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=64;")

# Files from source directories
file(GLOB main_sources "main/src/*cpp")

# Velo chain
file(GLOB velo_common "cuda/velo/common/src/*cu")
file(GLOB velo_clustering_sources "cuda/velo/mask_clustering/src/*cu")
file(GLOB velo_phi_and_sort "cuda/velo/calculate_phi_and_sort/src/*cu")
file(GLOB velo_search_by_triplet "cuda/velo/search_by_triplet/src/*cu")
file(GLOB velo_simplified_kalman_filter "cuda/velo/simplified_kalman_filter/src/*cu")
file(GLOB velo_consolidate_tracks "cuda/velo/consolidate_tracks/src/*cu")

# Stream
file(GLOB stream_handlers "stream/handlers/src/*cu")
file(GLOB stream_sequence "stream/sequence/src/*cu")

# x86 sources
file(GLOB x86_clustering "x86/velo/clustering/src/*cpp")

cuda_add_executable(cu_hlt ${main_sources} ${velo_common} ${velo_clustering_sources} ${velo_phi_and_sort} ${velo_search_by_triplet} ${velo_simplified_kalman_filter} ${velo_consolidate_tracks} ${stream_handlers} ${stream_sequence} ${x86_clustering})

target_link_libraries(cu_hlt tbb)

