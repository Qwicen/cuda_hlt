cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(Allen C CXX CUDA)

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

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")

# Configuration of CUDA compute architecture
set(CUDA_ARCH "MIN" CACHE STRING "Cuda architecture")

if (CUDA_ARCH STREQUAL "MIN" OR CUDA_ARCH STREQUAL "MAX" OR CUDA_ARCH STREQUAL "COMP")
  set(OUTPUTFILE ${PROJECT_BINARY_DIR}/cuda_arch) # No suffix required
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/cuda_arch.cu)
  execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  if(CUDA_ARCH STREQUAL "MAX")
    set(CHECK_ARGS "-l")
  endif()
  message(STATUS "Detecting ${CUDA_ARCH} CUDA architecture")
  execute_process(COMMAND ${OUTPUTFILE} ${CHECK_ARGS}
    RESULT_VARIABLE CUDA_RETURN_CODE
    OUTPUT_VARIABLE CUDA_ARCH_OUTPUT)

  if(${CUDA_RETURN_CODE} EQUAL 1)
    message(FATAL_ERROR "${CUDA_ARCH}")
  else()
    if (CUDA_ARCH STREQUAL "MIN")
      message(STATUS "Detecting ${CUDA_ARCH} CUDA architecture - sm_30")
      set(ARCH_FLAG "-arch=sm_30")
    else()
      message(STATUS "Detecting ${CUDA_ARCH} CUDA architecture - ${CUDA_ARCH_OUTPUT}")
      set(ARCH_FLAG "-arch=${CUDA_ARCH_OUTPUT}")
    endif()
  endif()
else()
  message(STATUS "Cuda architecture manually set to ${CUDA_ARCH}")
  set(ARCH_FLAG "-arch=${CUDA_ARCH}")
endif()

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

option(USE_KALMAN_SINGLE_PRECISION OFF)

# find_package(CUDA REQUIRED)
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
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG --generate-line-info ")
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --generate-line-info ")
elseif(${CMAKE_BUILD_TYPE} STREQUAL Debug)
  set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -g -DDEBUG ")
endif(${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)

option(CUDA_VEBOSE_BUILD OFF)
if (CUDA_VERBOSE_BUILD)
  set(CUDA_VERBOSE_FLAGS "--resource-usage --verbose --nvlink-options=--verbose -Xptxas=--verbose")
else()
  set(CUDA_VERBOSE_FLAGS "")
endif()
# In case we want to compile with clang:
# --compiler-bindir /cvmfs/sft.cern.ch/lcg/releases/clang/6.0.0-6647e/x86_64-centos7-gcc62-opt/bin/clang++;
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS " ${ARCH_FLAG} --use_fast_math --expt-relaxed-constexpr --maxrregcount=63 ${CUDA_VERBOSE_FLAGS}")

add_subdirectory(cuda)
add_subdirectory(stream)
add_subdirectory(checker)
add_subdirectory(x86/UT/PrVeloUT)
add_subdirectory(x86/SciFi)
add_subdirectory(x86/velo/clustering)
add_subdirectory(x86/PV/beamlinePV)
add_subdirectory(mdf)

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(main/include)
include_directories(x86/velo/clustering/include)
include_directories(cuda/global_event_cut/include)
include_directories(cuda/UT/common/include)
include_directories(cuda/UT/PrVeloUT/include)
include_directories(cuda/UT/compassUT/include)
include_directories(cuda/UT/UTDecoding/include)
include_directories(cuda/UT/consolidate/include)
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
include_directories(cuda/event_model/UT/include)
include_directories(cuda/event_model/SciFi/include)
include_directories(cuda/event_model/common/include)
include_directories(checker/tracking/include)
include_directories(checker/pv/include)
include_directories(stream/sequence/include)
include_directories(x86/SciFi/include)
include_directories(cuda/SciFi/PrForward/include)
include_directories(cuda/SciFi/consolidate/include)
include_directories(cuda/UT/UTDecoding/include)
include_directories(cuda/kalman/ParKalman/include)
include_directories(mdf/include)

file(GLOB common_sources "main/src/*")

# Remove main.cpp from common_sources
get_filename_component(main_cpp_path ${CMAKE_CURRENT_SOURCE_DIR}/main/src/main.cpp ABSOLUTE)
list(REMOVE_ITEM common_sources "${main_cpp_path}")

# common library
add_library(Common ${common_sources})
target_link_libraries(Common mdf)

set_property(TARGET Common PROPERTY
             CUDA_SEPARABLE_COMPILATION ON)

# main executable
add_executable(Allen ${main_cpp_path})
set_property(TARGET Allen
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Allen PRIVATE
  Stream
  TrackChecking
  PVChecking
  CheckClustering
)

if (ROOT_FOUND)
  target_compile_definitions(Allen PUBLIC WITH_ROOT)
  target_link_libraries(Allen PRIVATE ${ROOT_LIBRARIES})
endif()

option(BUILD_TESTS "Build test programs" OFF)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif(BUILD_TESTS)
