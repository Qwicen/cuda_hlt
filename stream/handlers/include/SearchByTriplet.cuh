#pragma once

#include "../../../cuda/velo/common/include/Definitions.cuh"
#include "../../../cuda/velo/search_by_triplet/include/SearchByTriplet.cuh"
#include "../../../main/include/Common.h"
#include <vector>
#include <iostream>

struct SearchByTriplet {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  uint32_t* dev_velo_cluster_container;
  uint* dev_module_cluster_start;
  uint* dev_module_cluster_num;
  Track* dev_tracks;
  Track* dev_tracklets;
  uint* dev_tracks_to_follow;
  uint* dev_weak_tracks;
  bool* dev_hit_used;
  int* dev_atomics_storage;
  short* dev_h0_candidates;
  short* dev_h2_candidates;
  unsigned short* dev_rel_indices;

  SearchByTriplet() = default;

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
    uint32_t* param_dev_velo_cluster_container,
    uint* param_dev_module_cluster_start,
    uint* param_dev_module_cluster_num,
    Track* param_dev_tracks,
    Track* param_dev_tracklets,
    uint* param_dev_tracks_to_follow,
    uint* param_dev_weak_tracks,
    bool* param_dev_hit_used,
    int* param_dev_atomics_storage,
    short* param_dev_h0_candidates,
    short* param_dev_h2_candidates,
    unsigned short* param_dev_rel_indices
  ) {
    dev_velo_cluster_container = param_dev_velo_cluster_container;
    dev_module_cluster_start = param_dev_module_cluster_start;
    dev_module_cluster_num = param_dev_module_cluster_num;
    dev_tracks = param_dev_tracks;
    dev_tracklets = param_dev_tracklets;
    dev_tracks_to_follow = param_dev_tracks_to_follow;
    dev_weak_tracks = param_dev_weak_tracks;
    dev_hit_used = param_dev_hit_used;
    dev_atomics_storage = param_dev_atomics_storage;
    dev_h0_candidates = param_dev_h0_candidates;
    dev_h2_candidates = param_dev_h2_candidates;
    dev_rel_indices = param_dev_rel_indices;
  }

  void operator()();

  void print_output(
    const uint number_of_events
  );
};
