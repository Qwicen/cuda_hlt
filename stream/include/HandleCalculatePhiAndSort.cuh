#pragma once

#include "Measurable.cuh"
#include "CalculatePhiAndSort.cuh"

struct CalculatePhiAndSort : public Measurable {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  char* dev_events;
  unsigned int* dev_event_offsets;
  unsigned int* dev_hit_offsets;
  float* dev_hit_phi;
  int32_t* dev_hit_temp;
  unsigned short* dev_hit_permutation;

  CalculatePhiAndSort() : Measurable() {}

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    char* param_dev_events,
    unsigned int* param_dev_event_offsets,
    unsigned int* param_dev_hit_offsets,
    float* param_dev_hit_phi,
    int32_t* param_dev_hit_temp,
    unsigned short* param_dev_hit_permutation
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
    dev_events = param_dev_events;
    dev_event_offsets = param_dev_event_offsets;
    dev_hit_offsets = param_dev_hit_offsets;
    dev_hit_phi = param_dev_hit_phi;
    dev_hit_temp = param_dev_hit_temp;
    dev_hit_permutation = param_dev_hit_permutation;
  }

  void operator()();
};
