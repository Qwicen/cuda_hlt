cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cu_hlt C CXX)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}) # for find_package
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) # for find_package

# Deal with build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)
MESSAGE(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

# Configured sequence
if(NOT SEQUENCE)
  set(SEQUENCE DefaultSequence)
endif(NOT SEQUENCE)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/stream/sequence_setup/include/sequences/${SEQUENCE}.cuh" "${CMAKE_CURRENT_SOURCE_DIR}/stream/sequence_setup/include/sequences/ConfiguredSequence.cuh" COPYONLY)
message(STATUS "Configured sequence: " ${SEQUENCE})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

find_package(ZLIB REQUIRED)

option(USE_LZMA OFF)
if(USE_LZMA)
  find_package(LibLZMA REQUIRED)
else(USE_LZMA)
  set(LZMA_FOUND OFF)
endif(USE_LZMA)

option(USE_LZ4 OFF)
if(USE_LZ4)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LZ4 REQUIRED liblz4)
else(USE_LZ4)
  set(LZ4_FOUND OFF)
endif(USE_LZ4)

find_package(CUDA REQUIRED)
#set(CUDA_HOST_COMPILER "g++")
set(CUDA_SEPARABLE_COMPILATION ON)
option(CUDA_PROPAGATE_HOST_FLAGS OFF)

option(USE_ROOT OFF)
if ((EXISTS $ENV{ROOTSYS}) AND (USE_ROOT))
   if(EXISTS $ENV{ROOTSYS}/cmake/ROOTConfig.cmake) # ROOT was compiled with cmake
     list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
   else() # ROOT was compiled with configure/make
     list(APPEND CMAKE_MODULE_PATH $ENV{ROOTSYS}/etc/cmake)
   endif()
   find_package( ROOT )
   if ( ROOT_FOUND )
      message(STATUS "Compiling with ROOT: " ${ROOT_INCLUDE_DIRS})
    else()
      message(STATUS "Compiling without ROOT")
   endif()
else()
   message(STATUS "Compiling without ROOT")
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
add_subdirectory(mdf)

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
include_directories(mdf/include)

file(GLOB common_sources "main/src/*")

# Remove main.cpp from common_sources
get_filename_component(main_cpp_path ${CMAKE_CURRENT_SOURCE_DIR}/main/src/main.cpp ABSOLUTE)
list(REMOVE_ITEM common_sources "${main_cpp_path}")

# common library
cuda_add_library(Common ${common_sources})
target_link_libraries(Common mdf)

# main executable
cuda_add_executable(cu_hlt ${main_cpp_path})

target_link_libraries(cu_hlt
  Common
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

if (ROOT_FOUND)
  target_compile_definitions(cu_hlt PUBLIC WITH_ROOT)
  target_link_libraries(cu_hlt ${ROOT_LIBRARIES})
endif()
