cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cuForward CXX)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11 -ltbb")

find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER "g++")

set(CUDA_SEPARABLE_COMPILATION ON)

# Options for P100:
# -std=c++11;--gpu-architecture=compute_60;--gpu-code=sm_60;--maxrregcount=64;

set(CUDA_NVCC_FLAGS "-std=c++11;-O3;-g;--use_fast_math;--expt-relaxed-constexpr;--generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=64;--gpu-architecture=compute_60;--gpu-code=sm_60;")

cuda_add_executable(cuForward main/src/main.cpp main/src/Timer.cpp main/src/Tools.cpp main/src/Logger.cpp cuda/src/TrackForwarding.cu cuda/src/TrackSeeding.cu cuda/src/FillCandidates.cu cuda/src/SearchByTriplet.cu cuda/src/ProcessModules.cu cuda/src/TrackSeedingFirst.cu cuda/src/WeakTracksAdder.cu cuda/src/CalculatePhi.cu cuda/src/SortByPhi.cu cuda/src/CalculatePhiAndSort.cu cuda/src/VeloKalmanFilter.cu stream/src/Stream.cu stream/src/CalculatePhiAndSort.cu stream/src/CalculateVeloStates.cu stream/src/SearchByTriplet.cu)
