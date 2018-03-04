#pragma once

#include "Measurable.cuh"

struct CalculatePhiAndSort : public Measurable {
  // Call options
  dim3 numBlocks, numThreads;

  // Cuda stream
  cudaStream_t& stream;

  // Call parameters
  char* dev_input;
  unsigned int* dev_event_offsets;
  unsigned int* dev_hit_offsets;
  float* dev_hit_phi;
  int32_t* dev_hit_temp;
  unsigned short* dev_hit_permutation;

  CalculatePhiAndSort(
    const dim3& numBlocks,
    const dim3& numThreads,
    cudaStream_t& stream,
    char* dev_input,
    unsigned int* dev_event_offsets,
    unsigned int* dev_hit_offsets,
    float* dev_hit_phi,
    int32_t* dev_hit_temp,
    unsigned short* dev_hit_permutation
  ) : numBlocks(numBlocks), numThreads(numThreads), stream(stream), dev_input(dev_input), 
    dev_event_offsets(dev_event_offsets), dev_hit_offsets(dev_hit_offsets),
    dev_hit_offsets(dev_hit_offsets), dev_hit_phi(dev_hit_phi), dev_hit_temp(dev_hit_temp) {
    Measurable();
  }

  void operator()() {
    calculatePhiAndSort<<<numBlocks, numThreads, 0, stream>>>(
      dev_input,
      dev_event_offsets,
      dev_hit_offsets,
      dev_hit_phi,
      dev_hit_temp,
      dev_hit_permutation
    );
  }
};
