#pragma once

#include "../../cuda/velo/mask_clustering/include/PrefixSum.cuh"

struct PrefixSum {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  unsigned int* dev_estimated_input_size;
  unsigned int array_size;

  PrefixSum() = default;

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    unsigned int* param_dev_estimated_input_size,
    unsigned int param_array_size
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
    dev_estimated_input_size = param_dev_estimated_input_size;
    array_size = param_array_size;
  }

  void operator()();
};
