#pragma once

#include "cuda_runtime.h"

struct Helper {
  /**
   * @brief Invokes an object with operator()
   *        If requested, stores the time it took
   */
  template<class T>
  static void invoke(
    T& t,
    const std::string& name,
    std::vector<std::pair<std::string, float>>& times,
    cudaEvent_t& cuda_event_start,
    cudaEvent_t& cuda_event_stop,
    const bool do_print_timing = false
  ) {
    if (!do_print_timing) {
      // Invoke algorithm
      t();
      cudaCheck(cudaPeekAtLastError());
    } else {
      float time;
      cudaEventRecord(cuda_event_start, *(t.stream));
      t();
      cudaEventRecord(cuda_event_stop, *(t.stream));
      cudaEventSynchronize(cuda_event_stop);
      cudaEventElapsedTime(&time, cuda_event_start, cuda_event_stop);
      cudaCheck(cudaPeekAtLastError());
      times.emplace_back(name, 0.001 * time);
    }
  }
};
