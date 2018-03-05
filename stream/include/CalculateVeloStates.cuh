#pragma once

#include "Measurable.cuh"
#include "../../cuda/include/VeloKalmanFilter.cuh"

struct CalculateVeloStates : public Measurable {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t& stream;

  // Call parameters
  char* dev_events;
  char* dev_consolidated_tracks;
  VeloState* dev_velo_states;
  int32_t* dev_hit_temp;
  unsigned int* dev_event_offsets;
  unsigned int* dev_hit_offsets;

  CalculateVeloStates(
    const dim3& num_blocks,
    const dim3& num_threads,
    cudaStream_t& stream,
    char* dev_events,
    char* dev_consolidated_tracks,
    VeloState* dev_velo_states,
    int32_t* dev_hit_temp,
    unsigned int* dev_event_offsets,
    unsigned int* dev_hit_offsets
  ) : num_blocks(num_blocks), num_threads(num_threads), stream(stream), dev_events(dev_events), 
    dev_consolidated_tracks(dev_consolidated_tracks), dev_velo_states(dev_velo_states),
    dev_hit_temp(dev_hit_temp), dev_event_offsets(dev_event_offsets),
    dev_hit_offsets(dev_hit_offsets) {
    Measurable();
  }

  void operator()() {
    velo_fit<<<num_blocks, num_threads, 0, stream>>>(
      dev_events,
      dev_consolidated_tracks,
      dev_velo_states,
      dev_hit_temp,
      dev_event_offsets,
      dev_hit_offsets
    );
  }
};
