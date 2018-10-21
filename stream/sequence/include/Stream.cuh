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
#include "DynamicScheduler.cuh"
#include "SequenceSetup.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "Constants.cuh"
#include "run_VeloUT_CPU.h"
#include "VeloEventModel.cuh"
#include "UTDefinitions.cuh"

#include "EstimateInputSize.cuh"

class Timer;

struct Stream {
  // Sequence and arguments
  sequence_t sequence;

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

  // Dynamic scheduler
  DynamicScheduler<algorithm_tuple_t, argument_tuple_t> scheduler;

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
    const RuntimeOptions& runtime_options
  );

  void print_timing(
    const uint number_of_events,
    const std::vector<std::pair<std::string, float>>& times
  );

  template<typename T>
  void visit(
    T& state,
    const int sequence_step,
    ArgumentManager<argument_tuple_t>& arguments,
    const std::tuple<const uint, const char*, const uint*, const size_t, const size_t>& const_arguments);
};
