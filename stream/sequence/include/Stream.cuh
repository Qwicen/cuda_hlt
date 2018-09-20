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
#include "Catboost.h"
#include "TestCatboost.h"
#include "BaseDynamicScheduler.cuh"
#include "SequenceSetup.cuh"
#include "PrVeloUTMagnetToolDefinitions.cuh"

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
  bool run_on_x86;

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
  int* host_atomics_veloUT;

  //Catboost
  int tree_num;
  int model_float_feature_num;
  int model_bin_feature_num;
  int* host_tree_sizes;
  int* host_border_nums;
  int** host_tree_splits;
  float* host_catboost_output;
  float** host_borders;
  float** host_features;
  double** host_leaf_values;
  const int* treeSplitsPtr_flat;
  const double* leafValuesPtr_flat;
  const NCatBoostFbs::TObliviousTrees* ObliviousTrees;

  // Dynamic scheduler
  BaseDynamicScheduler scheduler;

  // GPU pointers
  char* dev_velo_geometry;
  char* dev_base_pointer;
  PrUTMagnetTool* dev_ut_magnet_tool;
  
  // Monte Carlo folder name
  std::string folder_name_MC;
  uint start_event_offset;

  cudaError_t initialize(
    const std::vector<char>& velopix_geometry,
    const PrUTMagnetTool* host_ut_magnet_tool,
    const uint max_number_of_events,
    const bool param_transmit_device_to_host,
    const bool param_do_check,
    const bool param_do_simplified_kalman_filter,
    const bool param_print_memory_usage,
    const bool param_run_on_x86,
    const std::string& param_folder_name_MC,
    const uint param_start_event_offset,
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
