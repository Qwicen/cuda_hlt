#pragma once

#include "../../../cuda/velo/mask_clustering/include/MaskedVeloClustering.cuh"
#include "../../../main/include/Common.h"
#include "Measurable.cuh"
#include <vector>
#include <iostream>

struct MaskedVeloClustering : public Measurable {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  char* dev_raw_input;
  uint* dev_raw_input_offsets;
  uint* dev_module_cluster_start;
  uint* dev_module_cluster_num;
  uint* dev_module_candidate_num;
  uint* dev_cluster_candidates;
  uint32_t* dev_velo_cluster_container;
  char* dev_velo_geometry;

  MaskedVeloClustering() = default;

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    char* param_dev_raw_input,
    uint* param_dev_raw_input_offsets,
    uint* param_dev_module_cluster_start,
    uint* param_dev_module_cluster_num,
    uint* param_dev_module_candidate_num,
    uint* param_dev_cluster_candidates,
    uint32_t* param_dev_velo_cluster_container,
    char* param_dev_velo_geometry
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
    dev_raw_input = param_dev_raw_input,
    dev_raw_input_offsets = param_dev_raw_input_offsets,
    dev_module_cluster_start = param_dev_module_cluster_start;
    dev_module_cluster_num = param_dev_module_cluster_num;
    dev_module_candidate_num = param_dev_module_candidate_num;
    dev_cluster_candidates = param_dev_cluster_candidates;
    dev_velo_cluster_container = param_dev_velo_cluster_container;
    dev_velo_geometry = param_dev_velo_geometry;
  }

  void operator()();

  void print_output(
    const uint number_of_events,
    const int print_max_per_module = -1
  );
};
