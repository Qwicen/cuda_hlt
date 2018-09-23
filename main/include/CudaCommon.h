#pragma once

#include "cuda_runtime.h"
#include <stdexcept>

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
