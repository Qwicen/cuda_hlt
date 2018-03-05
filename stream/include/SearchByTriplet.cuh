#pragma once

#include "Measurable.cuh"
#include "../../cuda/include/Definitions.cuh"
#include "../../cuda/include/SearchByTriplet.cuh"

struct SearchByTriplet : public Measurable {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t& stream;

  // Call parameters
  Track* dev_tracks;
  char* dev_input;
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

  SearchByTriplet(
    const dim3& num_blocks,
    const dim3& num_threads,
    cudaStream_t& stream,
    Track* dev_tracks,
    char* dev_input,
    unsigned int* dev_tracks_to_follow,
    bool* dev_hit_used,
    int* dev_atomics_storage,
    Track* dev_tracklets,
    unsigned int* dev_weak_tracks,
    unsigned int* dev_event_offsets,
    unsigned int* dev_hit_offsets,
    short* dev_h0_candidates,
    short* dev_h2_candidates,
    unsigned short* dev_rel_indices,
    float* dev_hit_phi,
    int32_t* dev_hit_temp
  ) : num_blocks(num_blocks), num_threads(num_threads), stream(stream), dev_tracks(dev_tracks), 
    dev_input(dev_input), dev_tracks_to_follow(dev_tracks_to_follow),
    dev_hit_used(dev_hit_used), dev_atomics_storage(dev_atomics_storage),
    dev_tracklets(dev_tracklets), dev_weak_tracks(dev_weak_tracks),
    dev_event_offsets(dev_event_offsets), dev_hit_offsets(dev_hit_offsets),
    dev_h0_candidates(dev_h0_candidates), dev_h2_candidates(dev_h2_candidates),
    dev_rel_indices(dev_rel_indices), dev_hit_phi(dev_hit_phi),
    dev_hit_temp(dev_hit_temp) {
    Measurable();
  }

  void operator()() {
    searchByTriplet<<<num_blocks, num_threads, 0, stream>>>(
      dev_tracks,
      dev_input,
      dev_tracks_to_follow,
      dev_hit_used,
      dev_atomics_storage,
      dev_tracklets,
      dev_weak_tracks,
      dev_event_offsets,
      dev_hit_offsets,
      dev_h0_candidates,
      dev_h2_candidates,
      dev_rel_indices,
      dev_hit_phi,
      dev_hit_temp
    );
  }
};
