#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include "../../main/include/Common.h"
#include "../../main/include/Logger.h"
#include "../../main/include/Timer.h"
#include "../../cuda/include/Definitions.cuh"
#include "../include/CalculatePhiAndSort.cuh"
#include "../include/SearchByTriplet.cuh"
#include "../include/ConsolidateTracks.cuh"
#include "../include/Helper.cuh"

class Timer;

struct Stream {
  // Limiting constants for preallocation
  constexpr static unsigned int maximum_average_number_of_hits_per_event = TTF_MODULO;
  constexpr static unsigned int max_tracks_in_event = MAX_TRACKS;
  constexpr static unsigned int max_numhits_in_module = MAX_NUMHITS_IN_MODULE;
  constexpr static unsigned int atomic_space = NUM_ATOMICS + 1;
  // Stream datatypes
  cudaStream_t stream;
  cudaEvent_t cuda_generic_event;
  unsigned int stream_number;
  bool do_print_timing;
  bool perform_velo_kalman_filter;
  // Datatypes
  Track* dev_tracks;
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
  unsigned short* dev_hit_permutation;
  int* host_number_of_tracks_pinned;
  Track* host_tracks_pinned;
  // Resizeable datatype
  char* dev_events;
  size_t dev_events_size;
  // Algorithms
  CalculatePhiAndSort calculatePhiAndSort;
  SearchByTriplet searchByTriplet;
  ConsolidateTracks consolidateTracks;
  // Algorithm launch options
  dim3 num_blocks;
  dim3 consolidate_blocks;
  dim3 sort_num_threads;
  dim3 sbt_num_threads;
  dim3 consolidate_num_threads;
  // Launch tests
  bool transmit_host_to_device;
  bool transmit_device_to_host;

  Stream() = default;

  cudaError_t initialize(
    const std::vector<char>& events,
    const std::vector<unsigned int>& event_offsets,
    const std::vector<unsigned int>& hit_offsets,
    const unsigned int number_of_events,
    const size_t param_starting_events_size,
    const bool param_transmit_host_to_device,
    const bool param_transmit_device_to_host,
    const unsigned int param_stream_number = 0,
    const bool param_do_print_timing = true
  ) {
    cudaCheck(cudaStreamCreate(&stream));
    cudaCheck(cudaEventCreate(&cuda_generic_event));
    stream_number = param_stream_number;
    do_print_timing = param_do_print_timing;
    dev_events_size = param_starting_events_size;
    transmit_host_to_device = param_transmit_host_to_device;
    transmit_device_to_host = param_transmit_device_to_host;

    // Blocks and threads for each algorithm
    num_blocks = dim3(number_of_events);
    consolidate_blocks = dim3(number_of_events);
    sort_num_threads = dim3(64);
    sbt_num_threads = dim3(NUMTHREADS_X);
    consolidate_num_threads = dim3(32);

    // Do memory allocations only once
    // phi and sort
    cudaCheck(cudaMalloc((void**)&dev_events, dev_events_size));
    cudaCheck(cudaMalloc((void**)&dev_event_offsets, number_of_events * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&dev_hit_offsets, (number_of_events + 1) * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&dev_hit_phi, maximum_average_number_of_hits_per_event * number_of_events * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dev_hit_temp, maximum_average_number_of_hits_per_event * number_of_events * sizeof(int32_t)));
    cudaCheck(cudaMalloc((void**)&dev_hit_permutation, maximum_average_number_of_hits_per_event * number_of_events * sizeof(unsigned short)));
    // sbt
    cudaCheck(cudaMalloc((void**)&dev_tracks, number_of_events * max_tracks_in_event * sizeof(Track)));
    cudaCheck(cudaMalloc((void**)&dev_tracks_to_follow, number_of_events * TTF_MODULO * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&dev_hit_used, maximum_average_number_of_hits_per_event * number_of_events * sizeof(bool)));
    cudaCheck(cudaMalloc((void**)&dev_atomics_storage, number_of_events * atomic_space * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&dev_tracklets, maximum_average_number_of_hits_per_event * number_of_events * sizeof(Track)));
    cudaCheck(cudaMalloc((void**)&dev_weak_tracks, maximum_average_number_of_hits_per_event * number_of_events * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&dev_h0_candidates, 2 * maximum_average_number_of_hits_per_event * number_of_events * sizeof(short)));
    cudaCheck(cudaMalloc((void**)&dev_h2_candidates, 2 * maximum_average_number_of_hits_per_event * number_of_events * sizeof(short)));
    cudaCheck(cudaMalloc((void**)&dev_rel_indices, number_of_events * max_numhits_in_module * sizeof(unsigned short)));

    // Memory allocations for host memory (copy back)
    cudaCheck(cudaMallocHost((void**)&host_number_of_tracks_pinned, number_of_events * sizeof(int)));
    cudaCheck(cudaMallocHost((void**)&host_tracks_pinned, number_of_events * max_tracks_in_event * sizeof(Track)));

    // Prepare data (for tests)
    cudaCheck(cudaMemcpyAsync(dev_events, events.data(), events.size(), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(dev_event_offsets, event_offsets.data(), event_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(dev_hit_offsets, hit_offsets.data(), hit_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

    // Prepare kernels
    calculatePhiAndSort.set(
      num_blocks,
      sort_num_threads,
      stream,
      dev_events,
      dev_event_offsets,
      dev_hit_offsets,
      dev_hit_phi,
      dev_hit_temp,
      dev_hit_permutation
    );

    searchByTriplet.set(
      num_blocks,
      sbt_num_threads,
      stream,
      dev_tracks,
      dev_events,
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

    consolidateTracks.set(
      consolidate_blocks,
      consolidate_num_threads,
      stream,
      dev_atomics_storage,
      dev_tracks,
      dev_tracklets,
      dev_hit_offsets,
      dev_hit_permutation
    );

    return cudaSuccess;
  }

  ~Stream() {
    // // Free buffers
    // cudaCheck(cudaFree(dev_hit_permutation));
    // cudaCheck(cudaFree(dev_tracks_to_follow));
    // cudaCheck(cudaFree(dev_hit_used));
    // cudaCheck(cudaFree(dev_tracklets));
    // cudaCheck(cudaFree(dev_weak_tracks));
    // cudaCheck(cudaFree(dev_h0_candidates));
    // cudaCheck(cudaFree(dev_h2_candidates));
    // cudaCheck(cudaFree(dev_rel_indices));
    // cudaCheck(cudaFree(dev_event_offsets));
    // cudaCheck(cudaFree(dev_hit_offsets));
    // cudaCheck(cudaFree(dev_events));
    // cudaCheck(cudaFree(dev_hit_temp));
    // cudaCheck(cudaFree(dev_atomics_storage));
    // cudaCheck(cudaFree(dev_tracks));

    // cudaCheck(cudaStreamDestroy(stream));
    // cudaCheck(cudaStreamDestroy(stream_receive));
  }
  
  cudaError_t operator()(
    const char* host_events_pinned,
    const unsigned int* host_event_offsets_pinned,
    const unsigned int* host_hit_offsets_pinned,
    size_t host_events_pinned_size,
    size_t host_event_offsets_pinned_size,
    size_t host_hit_offsets_pinned_size,
    unsigned int start_event,
    unsigned int number_of_events,
    unsigned int number_of_repetitions
  );

  void print_timing(
    const unsigned int number_of_events,
    const std::vector<std::pair<std::string, float>>& times
  );
};
