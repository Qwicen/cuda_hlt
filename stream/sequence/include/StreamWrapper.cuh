#pragma once

#include <vector>
#include <string>
#include <stdint.h>

#include "VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.cuh"
#include "Logger.h"
#include "Common.h"

// Forward definition of Stream, to avoid
// inability to compile kernel calls (due to <<< >>>
// operators) from main.cpp
// 
// Note: main.cu wouldn't work due to nvcc not
//       supporting properly tbb (or the other way around).
struct Stream;

struct StreamWrapper {
  // Note: We need Stream* here due to the compiler
  //       needing to know the size of the allocated object
  std::vector<Stream*> streams;

  StreamWrapper() = default;

  ~StreamWrapper();

  /**
   * @brief Initializes n streams
   */
  void initialize_streams(
    const uint n,
    const std::vector<char>& velopix_geometry,
    const std::vector<char>& ut_boards,
    const std::vector<char>& ut_geometry,
    const PrUTMagnetTool* host_ut_magnet_tool,
    const uint number_of_events,
    const bool transmit_device_to_host,
    const bool do_check,
    const bool do_simplified_kalman_filter,
    const bool print_memory_usage,
    const bool run_on_x86,
    const std::string& folder_name_MC,
    const uint start_event_offset,
    const size_t reserve_mb
  );

  /**
   * @brief Runs stream i
   */
  void run_stream(
    const uint i,
    char* host_velopix_events,
    uint* host_velopix_event_offsets,
    const size_t velopix_events_size,
    const size_t velopix_event_offsets_size,
    char* host_ut_events,
    uint* host_ut_event_offsets,
    const size_t ut_events_size,
    const size_t ut_event_offsets_size,
    VeloUTTracking::HitsSoA *host_ut_hits_events,
    const PrUTMagnetTool* host_ut_magnet_tool,
    const uint number_of_events,
    const uint number_of_repetitions
  );
};
