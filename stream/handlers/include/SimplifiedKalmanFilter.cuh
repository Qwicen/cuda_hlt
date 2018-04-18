#pragma once

#include "../../../cuda/velo/simplified_kalman_filter/include/VeloKalmanFilter.cuh"
#include "Measurable.cuh"

struct SimplifiedKalmanFilter : public Measurable {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  uint32_t* dev_velo_cluster_container;
  uint* dev_module_cluster_start;
  int* dev_atomics_storage;
  Track* dev_tracks;
  VeloState* dev_velo_states;
  bool is_consolidated;

  SimplifiedKalmanFilter() : Measurable() {}

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    uint32_t* param_dev_velo_cluster_container,
    uint* param_dev_module_cluster_start,
    int* param_dev_atomics_storage,
    Track* param_dev_tracks,
    VeloState* param_dev_velo_states,
    bool param_is_consolidated
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
    dev_velo_cluster_container = param_dev_velo_cluster_container;
    dev_module_cluster_start = param_dev_module_cluster_start;
    dev_atomics_storage = param_dev_atomics_storage;
    dev_tracks = param_dev_tracks;
    dev_velo_states = param_dev_velo_states;
    is_consolidated = param_is_consolidated;
  }

  void operator()();
};
