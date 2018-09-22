#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <tuple>

#include "Common.h"
#include "CudaCommon.h"
#include "Logger.h"
#include "Timer.h"
#include "Tools.h"
#include "BaseDynamicScheduler.cuh"
#include "SequenceSetup.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "Constants.cuh"
#include "run_VeloUT_CPU.h"
#include "run_Forward_CPU.h"
#include "VeloEventModel.cuh"
#include "UTDefinitions.cuh"

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
  bool do_check;
  bool do_simplified_kalman_filter;
  bool do_print_memory_manager;
  bool run_on_x86;

  /* Pinned host datatypes */
  // Velo
  uint* host_velo_tracks_atomics;
  uint* host_velo_track_hit_number;
  char* host_velo_track_hits;
  uint* host_total_number_of_velo_clusters;
  uint* host_number_of_reconstructed_velo_tracks;
  uint* host_accumulated_number_of_hits_in_velo_tracks;
  char* host_velo_states;

  // UT
  uint* host_accumulated_number_of_ut_hits;
  uint* host_ut_hit_count;
  VeloUTTracking::TrackUT* host_veloUT_tracks;
  int* host_atomics_veloUT;
  UTHits * host_ut_hits_decoded;
 
  // SciFi 
  uint* host_accumulated_number_of_scifi_hits;
  SciFi::HitsSoA* host_scifi_hits;
  SciFi::Track* host_scifi_tracks;
  uint* host_n_scifi_tracks;
  
  

  // Dynamic scheduler
  BaseDynamicScheduler scheduler;

  // GPU pointers
  char* dev_velo_geometry;
  char* dev_ut_boards;
  char* dev_ut_geometry;
  char* dev_scifi_geometry;
  char* dev_base_pointer;
  PrUTMagnetTool* dev_ut_magnet_tool;

  // Monte Carlo folder name
  std::string folder_name_MC;
  uint start_event_offset;

  // Constants
  Constants constants;

  cudaError_t initialize(
    const std::vector<char>& velopix_geometry,
    const std::vector<char>& ut_boards,
    const std::vector<char>& ut_geometry,
    const std::vector<char>& ut_magnet_tool,
    const std::vector<char>& scifi_geometry,
    const uint max_number_of_events,
    const bool param_do_check,
    const bool param_do_simplified_kalman_filter,
    const bool param_print_memory_usage,
    const bool param_run_on_x86,
    const std::string& param_folder_name_MC,
    const uint param_start_event_offset,
    const size_t param_reserve_mb,
    const uint param_stream_number,
    const Constants& param_constants
  );

  cudaError_t run_sequence(
    const uint i_stream,
    const char* host_velopix_events,
    const uint* host_velopix_event_offsets,
    const size_t host_velopix_events_size,
    const size_t host_velopix_event_offsets_size,
    const char* host_ut_events,
    const uint* host_ut_event_offsets,
    const size_t host_ut_events_size,
    const size_t host_ut_event_offsets_size,
    SciFi::HitsSoA *hits_layers_events_scifi,
    char* host_scifi_events,
    uint* host_scifi_event_offsets,
    const size_t scifi_events_size,
    const size_t scifi_event_offsets_size,
    const uint number_of_events,
    const uint number_of_repetitions
  );

  void print_timing(
    const uint number_of_events,
    const std::vector<std::pair<std::string, float>>& times
  );
};
