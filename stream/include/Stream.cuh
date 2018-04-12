#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "../../main/include/Common.h"
#include "../../main/include/Logger.h"
#include "../../main/include/Timer.h"
#include "../../cuda/velo/common/include/Definitions.cuh"
#include "../include/EstimateInputSize.cuh"
#include "../include/PrefixSum.cuh"
#include "../include/MaskedVeloClustering.cuh"
#include "../include/CalculatePhiAndSort.cuh"
#include "../include/SearchByTriplet.cuh"
#include "../include/ConsolidateTracks.cuh"
#include "../../x86/include/ClusteringCommon.h"

class Timer;

struct Stream {
  // Limiting constants for preallocation
  constexpr static uint maximum_average_number_of_hits_per_event = TTF_MODULO;
  constexpr static uint max_tracks_in_event = MAX_TRACKS;
  constexpr static uint max_numhits_in_module = MAX_NUMHITS_IN_MODULE;
  constexpr static uint atomic_space = NUM_ATOMICS + 1;
  // Stream datatypes
  cudaStream_t stream;
  cudaEvent_t cuda_generic_event;
  uint stream_number;
  bool perform_velo_kalman_filter;
  // Datatypes
  Track* dev_tracks;
  uint* dev_tracks_to_follow;
  bool* dev_hit_used;
  int* dev_atomics_storage;
  Track* dev_tracklets;
  uint* dev_weak_tracks;
  uint* dev_event_offsets;
  uint* dev_hit_offsets;
  short* dev_h0_candidates;
  short* dev_h2_candidates;
  unsigned short* dev_rel_indices;
  float* dev_hit_phi;
  int32_t* dev_hit_temp;
  unsigned short* dev_hit_permutation;
  int* host_number_of_tracks_pinned;
  Track* host_tracks_pinned;
  // Clustering input
  char* dev_raw_input;
  uint* dev_raw_input_offsets;
  uint* dev_estimated_input_size;
  uint* dev_module_cluster_num;
  uint* dev_module_candidate_num;
  uint* dev_cluster_candidates;
  uint32_t* dev_velo_cluster_container;
  unsigned char* dev_sp_patterns;
  unsigned char* dev_sp_sizes;
  float* dev_sp_fx;
  float* dev_sp_fy;
  char* dev_velo_geometry;
  // Resizeable datatype
  char* dev_events;
  size_t dev_events_size;
  // Algorithms
  EstimateInputSize estimateInputSize;
  PrefixSum prefixSum;
  MaskedVeloClustering maskedVeloClustering;
  CalculatePhiAndSort calculatePhiAndSort;
  SearchByTriplet searchByTriplet;
  ConsolidateTracks consolidateTracks;
  // Algorithm launch options
  dim3 num_blocks;
  dim3 estimate_input_size_blocks;
  dim3 prefix_sum_blocks;
  dim3 masked_velo_clustering_blocks;
  dim3 consolidate_blocks;
  // num threads
  dim3 estimate_input_size_threads;
  dim3 prefix_sum_threads;
  dim3 masked_velo_clustering_threads;
  dim3 sort_num_threads;
  dim3 sbt_num_threads;
  dim3 consolidate_num_threads;
  // Launch tests
  bool transmit_host_to_device;
  bool transmit_device_to_host;
  // Varying cluster container size
  uint velo_cluster_container_size;

  Stream() = default;

  cudaError_t initialize(
    const std::vector<char>& raw_events,
    const std::vector<uint>& event_offsets,
    const std::vector<uint>& hit_offsets,
    const std::vector<char>& geometry,
    const uint number_of_events,
    const size_t param_starting_events_size,
    const bool param_transmit_host_to_device,
    const bool param_transmit_device_to_host,
    const uint param_stream_number = 0
  );

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
    const uint* host_event_offsets_pinned,
    const uint* host_hit_offsets_pinned,
    size_t host_events_pinned_size,
    size_t host_event_offsets_pinned_size,
    size_t host_hit_offsets_pinned_size,
    uint start_event,
    uint number_of_events,
    uint number_of_repetitions
  );
};
