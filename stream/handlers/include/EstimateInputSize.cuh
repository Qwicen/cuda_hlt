#pragma once

#include "../../../cuda/velo/mask_clustering/include/EstimateInputSize.cuh"
#include "Measurable.cuh"

struct EstimateInputSize : public Measurable {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  char* dev_raw_input;
  uint* dev_raw_input_offsets;
  uint* dev_estimated_input_size;
  uint* dev_module_cluster_num;
  uint* dev_module_candidate_num;
  uint32_t* dev_cluster_candidates;

  EstimateInputSize() = default;

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    char* param_dev_raw_input,
    uint* param_dev_raw_input_offsets,
    uint* param_dev_estimated_input_size,
    uint* param_dev_module_cluster_num,
    uint* param_dev_module_candidate_num,
    uint32_t* param_dev_cluster_candidates
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
    dev_raw_input = param_dev_raw_input;
    dev_raw_input_offsets = param_dev_raw_input_offsets;
    dev_estimated_input_size = param_dev_estimated_input_size;
    dev_module_cluster_num = param_dev_module_cluster_num;
    dev_module_candidate_num = param_dev_module_candidate_num;
    dev_cluster_candidates = param_dev_cluster_candidates;
  }

  void operator()();
};
