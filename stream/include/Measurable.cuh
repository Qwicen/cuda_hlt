#pragma once

#include "Common.h"
#include "cuda_runtime.h"

/**
 * @brief Measurable class with timers
 */
struct Measurable {
  cudaEvent_t event_start, event_stop;
  Measurable() {
    cudaCheck(cudaEventCreate(&event_start));
    cudaCheck(cudaEventCreate(&event_stop));
  }
  ~Measurable() {
    // cudaCheck(cudaEventDestroy(event_start));
    // cudaCheck(cudaEventDestroy(event_stop));
  }
};
