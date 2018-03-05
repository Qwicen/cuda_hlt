#pragma once

#include "cuda_runtime.h"

struct Helper {
  /**
   * @brief Invokes an object with operator()
   *        and returns the timer
   */
  template<class T>
  static float invoke(T& t) {
    float time;
    cudaEventRecord(t.event_start, t.stream);
    // Invoke operator()
    t();
    cudaEventRecord(t.event_stop, t.stream);
    cudaEventSynchronize(t.event_stop);
    cudaEventElapsedTime(&time, t.event_start, t.event_stop);
    cudaCheck(cudaPeekAtLastError());
    return time;
  }
};
