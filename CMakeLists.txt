cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cu_hlt CXX)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

# Deal with build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)
MESSAGE(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

# option to do MC truth matching
if(NOT MC_CHECK)
  set(MC_CHECK Off)
endif(NOT MC_CHECK)
if(${MC_CHECK} STREQUAL On OR ${MC_CHECK} STREQUAL ON)
  MESSAGE(STATUS "MC_CHECK: On")
  add_definitions( -DMC_CHECK )
else(${MC_CHECK} STREQUAL On OR ${MC_CHECK} STREQUAL ON)
  MESSAGE(STATUS "MC_CHECK: Off")
endif(${MC_CHECK} STREQUAL On OR ${MC_CHECK} STREQUAL ON)

find_package(CUDA REQUIRED)
set(CUDA_HOST_COMPILER "g++")
set(CUDA_SEPARABLE_COMPILATION ON)
option(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Cuda: Deal with build type
if(${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)
  set(CUDA_BUILD_TYPE "-O3\;-g\;-DNDEBUG\;")
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
  set(CUDA_BUILD_TYPE "-O3\;-DNDEBUG\;")
elseif(${CMAKE_BUILD_TYPE} STREQUAL Debug)
  set(CUDA_BUILD_TYPE "-O0\;-G\;-g\;-DDEBUG\;")
endif(${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)

set(CUDA_NVCC_FLAGS "${CUDA_BUILD_TYPE}--use_fast_math;--expt-relaxed-constexpr;--generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=64;")

add_subdirectory(cuda)
add_subdirectory(stream)
add_subdirectory(x86/velo/clustering)
add_subdirectory(checker)
add_subdirectory(PrVeloUT)

# Files from source directories
file(GLOB main_sources "main/src/*cpp")

cuda_add_executable(cu_hlt ${main_sources})

target_link_libraries(cu_hlt tbb Velo Stream x86Clustering TrackChecking CheckClustering)
