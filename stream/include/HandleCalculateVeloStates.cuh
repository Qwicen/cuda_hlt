#pragma once

#include "Measurable.cuh"
#include "VeloKalmanFilter.cuh"

struct CalculateVeloStates : public Measurable {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  char* dev_events;
  int* dev_atomics_storage;
  Track* dev_tracks;
  VeloState* dev_velo_states;
  int32_t* dev_hit_temp;
  unsigned int* dev_event_offsets;
  unsigned int* dev_hit_offsets;

  CalculateVeloStates() : Measurable() {}

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    char* param_dev_events,
    int* param_dev_atomics_storage,
    Track* param_dev_tracks,
    VeloState* param_dev_velo_states,
    int32_t* param_dev_hit_temp,
    unsigned int* param_dev_event_offsets,
    unsigned int* param_dev_hit_offsets
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
    dev_events = param_dev_events;
    dev_atomics_storage = param_dev_atomics_storage;
    dev_tracks = param_dev_tracks;
    dev_velo_states = param_dev_velo_states;
    dev_hit_temp = param_dev_hit_temp;
    dev_event_offsets = param_dev_event_offsets;
    dev_hit_offsets = param_dev_hit_offsets;
  }

  void operator()();
};
