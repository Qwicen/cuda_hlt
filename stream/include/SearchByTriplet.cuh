#pragma once

#include "../../cuda/velo/common/include/Definitions.cuh"
#include "../../cuda/velo/search_by_triplet/include/SearchByTriplet.cuh"

struct SearchByTriplet {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  Track* dev_tracks;
  char* dev_events;
  unsigned int* dev_tracks_to_follow;
  bool* dev_hit_used;
  int* dev_atomics_storage;
  Track* dev_tracklets;
  unsigned int* dev_weak_tracks;
  unsigned int* dev_event_offsets;
  unsigned int* dev_hit_offsets;
  short* dev_h0_candidates;
  short* dev_h2_candidates;
  unsigned short* dev_rel_indices;
  float* dev_hit_phi;
  int32_t* dev_hit_temp;

  SearchByTriplet() = default;

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    Track* param_dev_tracks,
    char* param_dev_events,
    unsigned int* param_dev_tracks_to_follow,
    bool* param_dev_hit_used,
    int* param_dev_atomics_storage,
    Track* param_dev_tracklets,
    unsigned int* param_dev_weak_tracks,
    unsigned int* param_dev_event_offsets,
    unsigned int* param_dev_hit_offsets,
    short* param_dev_h0_candidates,
    short* param_dev_h2_candidates,
    unsigned short* param_dev_rel_indices,
    float* param_dev_hit_phi,
    int32_t* param_dev_hit_temp
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
    dev_tracks = param_dev_tracks;
    dev_events = param_dev_events;
    dev_tracks_to_follow = param_dev_tracks_to_follow;
    dev_hit_used = param_dev_hit_used;
    dev_atomics_storage = param_dev_atomics_storage;
    dev_tracklets = param_dev_tracklets;
    dev_weak_tracks = param_dev_weak_tracks;
    dev_event_offsets = param_dev_event_offsets;
    dev_hit_offsets = param_dev_hit_offsets;
    dev_h0_candidates = param_dev_h0_candidates;
    dev_h2_candidates = param_dev_h2_candidates;
    dev_rel_indices = param_dev_rel_indices;
    dev_hit_phi = param_dev_hit_phi;
    dev_hit_temp = param_dev_hit_temp;
  }

  void operator()();
};
