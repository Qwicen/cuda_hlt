#pragma once

#include <vector>
#include "TrackChecker.h"
#include "Logger.h"
#include "UTDefinitions.cuh"
#include "SciFiDefinitions.cuh"
#include "UTEventModel.cuh"
#include "MiniState.cuh"

/**
 * @brief Prepares tracks for Velo consolidated datatypes.
 */
std::vector<trackChecker::Tracks> prepareVeloTracks(
  const uint* track_atomics,
  const uint* track_hit_number,
  const char* track_hits,
  const uint number_of_events);

/**
 * @brief Prepares tracks for Velo, UT consolidated datatypes.
 */
std::vector<trackChecker::Tracks> prepareUTTracks(
  const uint* velo_track_atomics,
  const uint* velo_track_hit_number,
  const char* velo_track_hits,
  const int* ut_track_atomics,
  const uint* ut_track_hit_number,
  const char* ut_track_hits,
  const uint* ut_track_velo_indices,
  const float* ut_qop,
  const uint number_of_events);

/**
 * @brief Prepares tracks for Velo, UT, SciFi consolidated datatypes.
 */
std::vector<trackChecker::Tracks> prepareSciFiTracks(
  const uint* velo_track_atomics,
  const uint* velo_track_hit_number,
  const char* velo_track_hits,
  const int* ut_track_atomics,
  const uint* ut_track_hit_number,
  const char* ut_track_hits,
  const uint* ut_track_velo_indices,
  const float* ut_qop,
  const int* scifi_track_atomics,
  const uint* scifi_track_hit_number,
  const char* scifi_track_hits,
  const uint* scifi_track_ut_indices,
  const float* scifi_qop,
  const MiniState* scifi_states,
  const char* host_scifi_geometry,
  const std::array<float, 9>& host_inv_clus_res,
  const uint number_of_events);
