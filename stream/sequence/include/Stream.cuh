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
#include "Constants.cuh"
#include "VeloEventModel.cuh"
#include "UTDefinitions.cuh"
#include "RuntimeOptions.h"
#include "EstimateInputSize.cuh"
#include "HostBuffers.cuh"
#include "SequenceVisitor.cuh"

class Timer;

struct Stream {
  // Sequence and arguments
  sequence_t sequence_tuple;

  // Stream datatypes
  cudaStream_t cuda_stream;
  cudaEvent_t cuda_generic_event;
  uint stream_number;

  // Launch options
  bool do_check;
  bool do_simplified_kalman_filter;
  bool do_print_memory_manager;
  bool run_on_x86;

  // Dynamic scheduler
  DynamicScheduler<sequence_t, argument_tuple_t> scheduler;

  // Host buffers
  HostBuffers host_buffers;

  // Monte Carlo folder name
  std::string folder_name_MC;
  uint start_event_offset;

  // GPU Memory base pointer
  char* dev_base_pointer;

  // Constants
  Constants constants;

  // Visitors for sequence algorithms
  SequenceVisitor sequence_visitor;

  cudaError_t initialize(
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

  void run_monte_carlo_test(
    const uint number_of_events_requested
  );
  

  cudaError_t run_sequence(
    const RuntimeOptions& runtime_options
  );
};
