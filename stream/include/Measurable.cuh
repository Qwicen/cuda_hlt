#pragma once

#include "../../main/include/Common.h"
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
    // This, for some reason, segfaults
    // cudaEventDestroy(event_start);
    // cudaEventDestroy(event_stop);
  }
};
