#pragma once

#include <vector>
#include <string>
#include <stdint.h>

#include "VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "Logger.h"
#include "Common.h"
#include "Constants.cuh"

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
    const std::vector<char>& ut_magnet_tool,
    const std::vector<char>& scifi_geometry,
    const uint number_of_events,
    const bool do_check,
    const bool do_simplified_kalman_filter,
    const bool print_memory_usage,
    const bool run_on_x86,
    const std::string& folder_name_MC,
    const uint start_event_offset,
    const size_t reserve_mb,
    const Constants& constants
  );

  /**
   * @brief Runs stream.
   */
  void run_stream(const RuntimeOptions& runtime_options);

  /**
   * @brief Runs Monte Carlo test. Stream must be run beforehand.
   */
  void run_monte_carlo_test(const RuntimeOptions& runtime_options)
};
