#pragma once

#include <vector>
#include <string>
#include <stdint.h>

#include "../../../cuda/veloUT/common/include/VeloUTDefinitions.cuh"
#include "../../../cuda/veloUT/PrVeloUT/include/PrVeloUTMagnetToolDefinitions.cuh"

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
    const PrUTMagnetTool* host_ut_magnet_tool,
    const uint number_of_events,
    const bool transmit_device_to_host,
    const bool do_check,
    const bool do_simplified_kalman_filter,
    const bool print_memory_usage,
    const std::string& folder_name_MC,
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
    VeloUTTracking::HitsSoA *host_ut_hits_events,
    const PrUTMagnetTool* host_ut_magnet_tool,
    const uint number_of_events,
    const uint number_of_repetitions
  );
};
