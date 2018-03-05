#pragma once

#include "Measurable.cuh"
#include "../../cuda/include/CalculatePhiAndSort.cuh"

struct CalculatePhiAndSort : public Measurable {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t& stream;

  // Call parameters
  char* dev_events;
  unsigned int* dev_event_offsets;
  unsigned int* dev_hit_offsets;
  float* dev_hit_phi;
  int32_t* dev_hit_temp;
  unsigned short* dev_hit_permutation;

  CalculatePhiAndSort(
    const dim3& num_blocks,
    const dim3& num_threads,
    cudaStream_t& stream,
    char* dev_events,
    unsigned int* dev_event_offsets,
    unsigned int* dev_hit_offsets,
    float* dev_hit_phi,
    int32_t* dev_hit_temp,
    unsigned short* dev_hit_permutation
  ) : num_blocks(num_blocks), num_threads(num_threads), stream(stream), dev_events(dev_events), 
    dev_event_offsets(dev_event_offsets), dev_hit_offsets(dev_hit_offsets),
    dev_hit_phi(dev_hit_phi), dev_hit_temp(dev_hit_temp),
    dev_hit_permutation(dev_hit_permutation) {
    Measurable();
  }

  void operator()() {
    calculatePhiAndSort<<<num_blocks, num_threads, 0, stream>>>(
      dev_events,
      dev_event_offsets,
      dev_hit_offsets,
      dev_hit_phi,
      dev_hit_temp,
      dev_hit_permutation
    );
  }
};
