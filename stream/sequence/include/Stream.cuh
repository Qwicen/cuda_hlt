#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "../../../main/include/Common.h"
#include "../../../main/include/Logger.h"
#include "../../../main/include/Timer.h"
#include "../../../cuda/velo/common/include/VeloDefinitions.cuh"
#include "../../handlers/include/EstimateInputSize.cuh"
#include "../../handlers/include/PrefixSum.cuh"
#include "../../handlers/include/MaskedVeloClustering.cuh"
#include "../../handlers/include/CalculatePhiAndSort.cuh"
#include "../../handlers/include/SearchByTriplet.cuh"
#include "../../handlers/include/ConsolidateTracks.cuh"
#include "../../handlers/include/SimplifiedKalmanFilter.cuh"
#include "../../handlers/include/Helper.cuh"

class Timer;

struct Stream {
  // Limiting constants for preallocation
  constexpr static uint average_number_of_hits_per_event = TTF_MODULO;
  constexpr static uint max_tracks_in_event = MAX_TRACKS;
  constexpr static uint max_numhits_in_module = MAX_NUMHITS_IN_MODULE;
  constexpr static uint atomic_space = NUM_ATOMICS + 1;
  // Stream datatypes
  cudaStream_t stream;
  cudaEvent_t cuda_generic_event;
  cudaEvent_t cuda_event_start;
  cudaEvent_t cuda_event_stop;
  uint stream_number;
  // Algorithms
  EstimateInputSize estimateInputSize;
  PrefixSum prefixSum;
  MaskedVeloClustering maskedVeloClustering;
  CalculatePhiAndSort calculatePhiAndSort;
  SearchByTriplet searchByTriplet;
  ConsolidateTracks consolidateTracks;
  SimplifiedKalmanFilter simplifiedKalmanFilter;
  // Launch options
  bool transmit_host_to_device;
  bool transmit_device_to_host;
  bool do_consolidate;
  bool do_simplified_kalman_filter;
  bool print_individual_rates;
  // Varying cluster container size
  uint velo_cluster_container_size;
  // Data back transmission
  int* host_number_of_tracks_pinned;
  Track* host_tracks_pinned;

  Stream() = default;

  cudaError_t initialize(
    const std::vector<char>& raw_events,
    const std::vector<uint>& event_offsets,
    const std::vector<char>& geometry,
    const uint number_of_events,
    const size_t param_starting_events_size,
    const bool param_transmit_host_to_device,
    const bool param_transmit_device_to_host,
    const bool param_do_consolidate,
    const bool param_do_simplified_kalman_filter,
    const bool param_print_individual_rates,
    const uint param_stream_number
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
    size_t host_events_pinned_size,
    size_t host_event_offsets_pinned_size,
    uint start_event,
    uint number_of_events,
    uint number_of_repetitions
  );

  void print_timing(
    const uint number_of_events,
    const std::vector<std::pair<std::string, float>>& times
  );
};
