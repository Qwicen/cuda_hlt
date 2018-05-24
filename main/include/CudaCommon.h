#pragma once

#include "cuda_runtime.h"
#include <stdexcept>

/// For sanity check of input
// DvB: not used anywhere, defined as N_MODULES in cuda/velo/common/include/VeloDefinitions.cuh
//#define NUMBER_OF_SENSORS 52

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
