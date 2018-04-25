#pragma once

#include "../../../cuda/velo/consolidate_tracks/include/ConsolidateTracks.cuh"

struct ConsolidateTracks {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  int* dev_atomics_storage;
  TrackHits* dev_tracks;
  Track* dev_output_tracks;
  uint32_t* dev_velo_cluster_container;
  uint* dev_module_cluster_start;
  uint* dev_module_cluster_num;

  ConsolidateTracks() = default;

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
  }

  void setParameters(
    int* param_dev_atomics_storage,
    TrackHits* param_dev_tracks,
    Track* param_dev_output_tracks,
    uint32_t* param_dev_velo_cluster_container,
    uint* param_dev_module_cluster_start,
    uint* param_dev_module_cluster_num
  ) {
    dev_atomics_storage = param_dev_atomics_storage;
    dev_tracks = param_dev_tracks;
    dev_output_tracks = param_dev_output_tracks;
    dev_velo_cluster_container = param_dev_velo_cluster_container;
    dev_module_cluster_start = param_dev_module_cluster_start;
    dev_module_cluster_num =  param_dev_module_cluster_num;

  }

  void operator()();
};
