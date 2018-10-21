#pragma once

#include <cstdint>
#include <cassert>
#include "ClusteringDefinitions.cuh"
#include "Handler.cuh"

__global__ void estimate_input_size(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  uint* dev_estimated_input_size,
  uint* dev_module_cluster_num,
  uint* dev_module_candidate_num,
  uint32_t* dev_cluster_candidates,
  uint8_t* dev_velo_candidate_ks
);

template<typename R, typename... T>
struct EstimateInputSize {
  Handler<0, R, T...> handler {estimate_input_size};

  void set_arguments(T... param_arguments) {
    handler.set_arguments(param_arguments...);
  }

  void set_opts(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    const unsigned param_shared_memory_size = 0
  ) {
    handler.set_opts(
      param_num_blocks,
      param_num_threads,
      param_stream,
      param_shared_memory_size);
  }

  void invoke() {
    handler.invoke();
  }
};

template<typename R, typename... T>
static EstimateInputSize<R, T...> estimate_input_size_t(R(f)(T...)) {
  return EstimateInputSize<R, T...>{f};
}
