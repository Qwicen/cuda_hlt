#pragma once

#include "Measurable.cuh"

struct CalculateVeloStates : public Measurable {
  // Call options
  dim3 numBlocks, numThreads;

  // Cuda stream
  cudaStream_t& stream;

  // Call parameters
  char* dev_input;
  char* dev_consolidated_tracks;
  VeloState* dev_velo_states;
  int32_t* dev_hit_temp;
  unsigned int* dev_event_offsets;
  unsigned int* dev_hit_offsets;

  CalculatePhiAndSort(
    const dim3& numBlocks,
    const dim3& numThreads,
    cudaStream_t& stream,
    char* dev_input,
    char* dev_consolidated_tracks,
    VeloState* dev_velo_states,
    int32_t* dev_hit_temp,
    unsigned int* dev_event_offsets,
    unsigned int* dev_hit_offsets
  ) : numBlocks(numBlocks), numThreads(numThreads), stream(stream), dev_input(dev_input), 
    dev_consolidated_tracks(dev_consolidated_tracks), dev_velo_states(dev_velo_states),
    dev_hit_temp(dev_hit_temp), dev_event_offsets(dev_event_offsets),
    dev_hit_offsets(dev_hit_offsets) {
    Measurable();
  }

  void operator()() {
    velo_fit<<<numBlocks, numThreads, 0, stream>>>(
      dev_input,
      dev_consolidated_tracks,
      dev_velo_states,
      dev_hit_temp,
      dev_event_offsets,
      dev_hit_offsets
    );
  }
};
