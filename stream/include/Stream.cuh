#pragma once

#include <iostream>
#include <vector>
#include "../../main/include/Common.h"
#include "../../main/include/Logger.h"
#include "../../main/include/Timer.h"
#include "../../cuda/include/Definitions.cuh"
#include "../include/CalculatePhiAndSort.cuh"
#include "../include/SearchByTriplet.cuh"
#include "../include/CalculateVeloStates.cuh"
#include "../include/Helper.cuh"

class Timer;

struct Stream {
  // Limiting constants for preallocation
  constexpr static unsigned int maximum_average_number_of_hits_per_event = 2000;
  constexpr static unsigned int max_tracks_in_event = 1000;
  constexpr static unsigned int max_numhits_in_module = 256;
  constexpr static unsigned int atomic_space = NUM_ATOMICS + 1;
  // Stream datatypes
  cudaStream_t stream;
  unsigned int stream_number;
  bool do_print_timing;
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
  VeloState* dev_velo_states;
  // Resizeable datatype
  char* dev_events;
  size_t dev_events_size;
  // Algorithms
  CalculatePhiAndSort calculatePhiAndSort;
  SearchByTriplet searchByTriplet;
  CalculateVeloStates calculateVeloStates;
  // Algorithm launch options
  dim3 num_blocks;
  dim3 sort_num_threads;
  dim3 sbt_num_threads;
  dim3 velo_states_num_threads;

  Stream(
    unsigned int number_of_events,
    size_t starting_events_size,
    unsigned int stream_number = 0,
    bool do_print_timing = true
  ) :
    stream_number(stream_number),
    do_print_timing(do_print_timing),
    dev_events_size(starting_events_size),
    calculatePhiAndSort(CalculatePhiAndSort(stream)),
    searchByTriplet(SearchByTriplet(stream)),
    calculateVeloStates(CalculateVeloStates(stream)) {
    cudaStreamCreate(&stream);
    
    // Blocks and threads for each algorithm
    num_blocks = dim3(number_of_events);
    sort_num_threads = dim3(64);
    sbt_num_threads = dim3(NUMTHREADS_X);
    velo_states_num_threads = dim3(1024);

    // Do memory allocations only once
    // phi and sort
    cudaCheckVoid(cudaMalloc((void**)&dev_events, dev_events_size));
    cudaCheckVoid(cudaMalloc((void**)&dev_event_offsets, number_of_events * sizeof(unsigned int)));
    cudaCheckVoid(cudaMalloc((void**)&dev_hit_offsets, (number_of_events + 1) * sizeof(unsigned int)));
    cudaCheckVoid(cudaMalloc((void**)&dev_hit_phi, maximum_average_number_of_hits_per_event * number_of_events * sizeof(float)));
    cudaCheckVoid(cudaMalloc((void**)&dev_hit_temp, maximum_average_number_of_hits_per_event * number_of_events * sizeof(int32_t)));
    cudaCheckVoid(cudaMalloc((void**)&dev_hit_permutation, maximum_average_number_of_hits_per_event * number_of_events * sizeof(unsigned short)));
    // sbt
    cudaCheckVoid(cudaMalloc((void**)&dev_tracks, number_of_events * max_tracks_in_event * sizeof(Track)));
    cudaCheckVoid(cudaMalloc((void**)&dev_tracks_to_follow, number_of_events * TTF_MODULO * sizeof(unsigned int)));
    cudaCheckVoid(cudaMalloc((void**)&dev_hit_used, maximum_average_number_of_hits_per_event * number_of_events * sizeof(bool)));
    cudaCheckVoid(cudaMalloc((void**)&dev_atomics_storage, number_of_events * atomic_space * sizeof(int)));
    cudaCheckVoid(cudaMalloc((void**)&dev_tracklets, maximum_average_number_of_hits_per_event * number_of_events * sizeof(Track)));
    cudaCheckVoid(cudaMalloc((void**)&dev_weak_tracks, maximum_average_number_of_hits_per_event * number_of_events * sizeof(unsigned int)));
    cudaCheckVoid(cudaMalloc((void**)&dev_h0_candidates, 2 * maximum_average_number_of_hits_per_event * number_of_events * sizeof(short)));
    cudaCheckVoid(cudaMalloc((void**)&dev_h2_candidates, 2 * maximum_average_number_of_hits_per_event * number_of_events * sizeof(short)));
    cudaCheckVoid(cudaMalloc((void**)&dev_rel_indices, number_of_events * max_numhits_in_module * sizeof(unsigned short)));
    // velo states
    cudaCheckVoid(cudaMalloc((void**)&dev_velo_states, number_of_events * max_tracks_in_event * STATES_PER_TRACK * sizeof(VeloState)));

    // Prepare kernels
    calculatePhiAndSort.set(
      num_blocks,
      sort_num_threads,
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

    calculateVeloStates.set(
      num_blocks,
      velo_states_num_threads,
      dev_events,
      dev_atomics_storage,
      dev_tracks,
      dev_velo_states,
      dev_hit_temp,
      dev_event_offsets,
      dev_hit_offsets
    );
  }

  ~Stream() {
    cudaStreamDestroy(stream);

    // Free buffers
    cudaCheckVoid(cudaFree(dev_hit_permutation));
    cudaCheckVoid(cudaFree(dev_tracks_to_follow));
    cudaCheckVoid(cudaFree(dev_hit_used));
    cudaCheckVoid(cudaFree(dev_tracklets));
    cudaCheckVoid(cudaFree(dev_weak_tracks));
    cudaCheckVoid(cudaFree(dev_h0_candidates));
    cudaCheckVoid(cudaFree(dev_h2_candidates));
    cudaCheckVoid(cudaFree(dev_rel_indices));
    cudaCheckVoid(cudaFree(dev_event_offsets));
    cudaCheckVoid(cudaFree(dev_hit_offsets));
    cudaCheckVoid(cudaFree(dev_events));
    cudaCheckVoid(cudaFree(dev_hit_temp));
    cudaCheckVoid(cudaFree(dev_atomics_storage));
    cudaCheckVoid(cudaFree(dev_tracks));
    cudaCheckVoid(cudaFree(dev_velo_states));
  }

  cudaError_t operator()(
    const std::vector<char>& events,
    const std::vector<unsigned int>& event_offsets,
    const std::vector<unsigned int>& hit_offsets,
    unsigned int start_event,
    unsigned int number_of_events,
    unsigned int number_of_repetitions
  );

  void print_timing(
    const unsigned int number_of_events,
    const std::vector<std::pair<std::string, float>>& times
  );
};
