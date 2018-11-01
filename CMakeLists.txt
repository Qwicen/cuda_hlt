cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cu_hlt C CXX)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package

# Deal with build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)
MESSAGE(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

#include_directories("/cvmfs/lhcb.cern.ch/lib/lcg/releases/ROOT/6.08.06-d7e12/x86_64-centos7-gcc62-opt/include/")
#include_directories("/cvmfs/lhcb.cern.ch/lib/lcg/releases/tbb/44_20160413-f254c/x86_64-centos7-gcc7-opt/include")
#include_directories("/usr/local/root/include")

find_package(CUDA REQUIRED)
#set(CUDA_HOST_COMPILER "g++")
set(CUDA_SEPARABLE_COMPILATION ON)
option(CUDA_PROPAGATE_HOST_FLAGS OFF)

if ( EXISTS $ENV{ROOTSYS} )
   if(EXISTS $ENV{ROOTSYS}/cmake/ROOTConfig.cmake) # ROOT was compiled with cmake
     list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
   else() # ROOT was compiled with configure/make
     list(APPEND CMAKE_MODULE_PATH $ENV{ROOTSYS}/etc/cmake)
   endif()
   find_package( ROOT )
   if ( ROOT_FOUND )
      message("-- Compiling with ROOT: ${ROOT_INCLUDE_DIRS}")
    else()
      message("-- Compiling without ROOT")
   endif()
else()
   message("-- Compiling without ROOT")
endif()

# Cuda: Deal with build type
if(${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)
  set(CUDA_BUILD_TYPE "-O3\;-g\;-DNDEBUG\;")
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
  set(CUDA_BUILD_TYPE "-O3\;-DNDEBUG\;")
elseif(${CMAKE_BUILD_TYPE} STREQUAL Debug)
  set(CUDA_BUILD_TYPE "-O0\;-G\;-g\;-DDEBUG\;")
endif(${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)

set(CUDA_NVCC_FLAGS "${CUDA_BUILD_TYPE}--use_fast_math;--expt-relaxed-constexpr;--generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=63;")

add_subdirectory(cuda)
add_subdirectory(stream)
add_subdirectory(checker)
add_subdirectory(x86/UT/PrVeloUT)
add_subdirectory(x86/SciFi)
add_subdirectory(x86/velo/clustering)

# Include directories
include_directories(main/include)
include_directories(x86/velo/clustering/include)
include_directories(cuda/UT/common/include)
include_directories(cuda/UT/PrVeloUT/include)
include_directories(cuda/UT/UTDecoding/include)
include_directories(cuda/velo/common/include)
include_directories(cuda/velo/calculate_phi_and_sort/include)
include_directories(cuda/velo/consolidate_tracks/include)
include_directories(cuda/velo/mask_clustering/include)
include_directories(cuda/velo/search_by_triplet/include)
include_directories(cuda/velo/simplified_kalman_filter/include)
include_directories(cuda/SciFi/common/include)
include_directories(cuda/muon/common/include)
include_directories(cuda/utils/prefix_sum/include)
include_directories(cuda/event_model/velo/include)
include_directories(cuda/event_model/common/include)
include_directories(checker/tracking/include)
include_directories(stream/sequence/include)
include_directories(x86/SciFi/include)
include_directories(cuda/SciFi/PrForward/include)
include_directories(cuda/UT/UTDecoding/include)
  
# Files from source directories
file(GLOB main_sources "main/src/*")

cuda_add_executable(cu_hlt ${main_sources})

if ( ROOT_FOUND )
  target_compile_definitions(cu_hlt PUBLIC WITH_ROOT)
  target_link_libraries(cu_hlt
    tbb
    Utils
    Velo
    UT
    SciFi
    x86VeloUT
    x86Forward
    Stream
    x86Clustering
    TrackChecking
    CheckClustering
    ${ROOT_LIBRARIES})
else()
  target_link_libraries(cu_hlt
    tbb
    Utils
    Velo
    UT
    SciFi
    x86VeloUT
    x86Forward
    Stream
    x86Clustering
    TrackChecking
    CheckClustering)
endif()
