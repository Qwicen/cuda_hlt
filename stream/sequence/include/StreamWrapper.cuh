#pragma once

#include <vector>
#include <string>
#include <stdint.h>

#include "UTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "SciFiDefinitions.cuh"
#include "Logger.h"
#include "Common.h"
#include "Constants.cuh"
#include "RuntimeOptions.h"

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
    const uint number_of_events,
    const bool print_memory_usage,
    const uint start_event_offset,
    const size_t reserve_mb,
    const Constants& constants);

  /**
   * @brief Runs stream.
   */
  void run_stream(const uint i, const RuntimeOptions& runtime_options);

  /**
   * @brief Runs Monte Carlo test. Stream must be run beforehand.
   */
  void run_monte_carlo_test(const uint i, const std::string& mc_folder, const uint number_of_events_requested);
};

/**
 * @brief Prints the configured sequence.
 *        Must be compiled by nvcc.
 */
void print_configured_sequence();
