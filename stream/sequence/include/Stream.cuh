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
#include "Constants.cuh"
#include "VeloEventModel.cuh"
#include "UTDefinitions.cuh"
#include "RuntimeOptions.h"
#include "EstimateInputSize.cuh"
#include "HostBuffers.cuh"
#include "SequenceVisitor.cuh"
#include "SchedulerMachinery.cuh"
#include "Scheduler.cuh"
#include "AlgorithmDependencies.cuh"
#include "CheckerInvoker.h"


class Timer;

struct Stream {
  using scheduler_t = Scheduler<configured_sequence_t, algorithms_dependencies_t, output_arguments_t>;
  using argument_manager_t = ArgumentManager<scheduler_t::arguments_tuple_t>;

  // Sequence and arguments
  configured_sequence_t sequence_tuple;

  // Stream datatypes
  cudaStream_t cuda_stream;
  cudaEvent_t cuda_generic_event;
  uint stream_number;

  // Launch options
  bool do_print_memory_manager;

  // Dynamic scheduler
  scheduler_t scheduler;

  // Host buffers
  HostBuffers host_buffers;

  // Start event offset
  uint start_event_offset;

  // GPU Memory base pointer
  char* dev_base_pointer;

  // Constants
  Constants constants;

  // Visitors for sequence algorithms
  SequenceVisitor sequence_visitor;

  cudaError_t initialize(
    const uint max_number_of_events,
    const bool param_print_memory_usage,
    const uint param_start_event_offset,
    const size_t param_reserve_mb,
    const uint param_stream_number,
    const Constants& param_constants
  );

  void run_monte_carlo_test(
    const std::string& mc_folder,
    const uint number_of_events_requested
  );

  cudaError_t run_sequence(
    const RuntimeOptions& runtime_options
  );
};
