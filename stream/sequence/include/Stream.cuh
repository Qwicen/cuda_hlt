#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <tuple>

#include "../../../main/include/Common.h"
#include "../../../main/include/CudaCommon.h"
#include "../../../main/include/Logger.h"
#include "../../../main/include/Timer.h"
#include "../../../main/include/Tools.h"
#include "../../scheduler/include/BaseDynamicScheduler.cuh"
#include "../../sequence_setup/include/SequenceSetup.cuh"
#include "run_VeloUT_CPU.h"
#include "run_PatPV_CPU.h"

class Timer;

struct Stream {
  // Sequence and arguments
  sequence_t sequence;
  argument_tuple_t arguments;

  // Sequence and argument names
  std::array<std::string, std::tuple_size<algorithm_tuple_t>::value> sequence_names;
  std::array<std::string, std::tuple_size<argument_tuple_t>::value> argument_names;

  // Stream datatypes
  cudaStream_t stream;
  cudaEvent_t cuda_generic_event;
  cudaEvent_t cuda_event_start;
  cudaEvent_t cuda_event_stop;
  uint stream_number;

  // Launch options
  bool transmit_host_to_device;
  bool transmit_device_to_host;
  bool do_check;
  bool do_simplified_kalman_filter;
  bool do_print_memory_manager;

  // Pinned host datatypes
  int* host_number_of_tracks;   //number of tracks in event
  int* host_accumulated_tracks;  //sum of tracks in previous events
  uint* host_velo_track_hit_number; // sum of hits in previous tracks in a event
  VeloTracking::Hit<mc_check_enabled>* host_velo_track_hits;  //array of all hits of all events
  uint* host_total_number_of_velo_clusters;
  uint* host_number_of_reconstructed_velo_tracks;  // total number of tracks in all events -> has only value at [0]
  uint* host_accumulated_number_of_hits_in_velo_tracks;  // total number of all hits in all events -> has only value at [0]
  VeloState* host_velo_states;  //velo state for each track -> one from simple least squares fit and one from simplified Kalman

  // Dynamic scheduler
  BaseDynamicScheduler scheduler;

  // GPU pointers
  char* dev_velo_geometry;
  char* dev_base_pointer;

  // Monte Carlo folder name
  std::string folder_name_MC;

  cudaError_t initialize(
    const std::vector<char>& raw_events,
    const std::vector<uint>& event_offsets,
    const std::vector<char>& geometry,
    const uint max_number_of_events,
    const bool param_transmit_host_to_device,
    const bool param_transmit_device_to_host,
    const bool param_do_check,
    const bool param_do_simplified_kalman_filter,
    const bool param_print_memory_usage,
    const std::string& param_folder_name_MC,
    const size_t param_reserve_mb,
    const uint param_stream_number
  );
  
  cudaError_t run_sequence(
    const uint i_stream,
    const char* host_events,
    const uint* host_event_offsets,
    const size_t host_events_size,
    const size_t host_event_offsets_size,
    VeloUTTracking::HitsSoA *hits_layers_events_ut,
    const uint32_t n_hits_layers_events_ut[][VeloUTTracking::n_layers],
    const uint number_of_events,
    const uint number_of_repetitions
  );

  void print_timing(
    const uint number_of_events,
    const std::vector<std::pair<std::string, float>>& times
  );
};
