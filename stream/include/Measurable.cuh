#pragma once

#include "cuda_runtime.h"

/**
 * @brief Measurable class with timers
 */
struct Measurable {
  cudaEvent_t event_start, event_stop;
  Measurable() {
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
  }
  ~Measurable() {
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
  }
};
