#pragma once

#include "cuda_runtime.h"
#include <stdexcept>
#include <iostream>

/**
 * @brief Macro to check cuda calls.
 */
#define cudaCheck(stmt) {                                \
  cudaError_t err = stmt;                                \
  if (err != cudaSuccess){                               \
    std::cerr << "Failed to run " << #stmt << std::endl; \
    std::cerr << cudaGetErrorString(err) << std::endl;   \
    throw std::invalid_argument("cudaCheck failed");     \
  }                                                      \
}

/**
 * @brief Cross architecture for statement.
 * @details It can be used to iterate with variable _TYPE _I from 0 through _END.
 */
#ifdef __CUDA_ARCH__
  #define FOR_STATEMENT(_TYPE, _I, _END) for (_TYPE _I=threadIdx.x; _I<_END; _I+=blockDim.x)
#else
  #define FOR_STATEMENT(_TYPE, _I, _END) for (_TYPE _I=0; _I<_END; ++_I)
#endif
