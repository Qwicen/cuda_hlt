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
#include "../../../cuda/veloUT/PrVeloUT/include/PrVeloUTMagnetToolDefinitions.cuh"

#include "run_VeloUT_CPU.h"

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
  int* host_number_of_tracks;
  int* host_accumulated_tracks;
  uint* host_velo_track_hit_number;
  VeloTracking::Hit<mc_check_enabled>* host_velo_track_hits;
  uint* host_total_number_of_velo_clusters;
  uint* host_number_of_reconstructed_velo_tracks;
  uint* host_accumulated_number_of_hits_in_velo_tracks;
  VeloState* host_velo_states;
  VeloUTTracking::TrackUT* host_veloUT_tracks;

  // Dynamic scheduler
  BaseDynamicScheduler scheduler;

  // GPU pointers
  char* dev_velo_geometry;
  char* dev_base_pointer;
  PrUTMagnetTool* dev_ut_magnet_tool;
  
  // Monte Carlo folder name
  std::string folder_name_MC;

  cudaError_t initialize(
    const std::vector<char>& velopix_geometry,
    const PrUTMagnetTool* host_ut_magnet_tool,
    const uint max_number_of_events,
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
    VeloUTTracking::HitsSoA *host_ut_hits_events,
    const PrUTMagnetTool* host_ut_magnet_tool,
    const uint number_of_events,
    const uint number_of_repetitions
  );

  void print_timing(
    const uint number_of_events,
    const std::vector<std::pair<std::string, float>>& times
  );
};
