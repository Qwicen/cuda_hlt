cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(SOAContainer CXX)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# C++11 without GNU extensions
# Requires CMAKE 3.1+ for Mac
if(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 3.1)
    add_compile_options(-std=c++11)
else()
    cmake_policy(VERSION 3.1) # Needed for Mac
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
# be moderately paranoid with flags
add_compile_options(-pedantic -Wall -Wextra)

# CMAKE 2
include_directories(include)
add_compile_options(-g -O2 -ffast-math -ftree-vectorize -march=native)

file(GLOB headers include/*)
