#pragma once

#include "Measurable.cuh"
#include "../../cuda/include/ConsolidateTracks.cuh"

struct ConsolidateTracks : public Measurable {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  int* dev_atomics_storage;
  Track* dev_tracks;
  Track* dev_output_tracks;
  unsigned int* dev_hit_offsets;
  unsigned short* dev_hit_permutation;

  ConsolidateTracks() : Measurable() {}

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    int* param_dev_atomics_storage,
    Track* param_dev_tracks,
    Track* param_dev_output_tracks,
    unsigned int* param_dev_hit_offsets,
    unsigned short* param_dev_hit_permutation
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
    dev_atomics_storage = param_dev_atomics_storage;
    dev_tracks = param_dev_tracks;
    dev_output_tracks = param_dev_output_tracks;
    dev_hit_offsets = param_dev_hit_offsets;
    dev_hit_permutation = param_dev_hit_permutation;
  }

  void operator()();
};
