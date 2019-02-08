#pragma once

#include <vector>
#include "Tracks.h"
#include "Logger.h"
#include "InputTools.h"
#include "UTDefinitions.cuh"
#include "SciFiDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "UTEventModel.cuh"
#include "SciFiEventModel.cuh"
#include "VeloConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "SciFiConsolidated.cuh"
#include "ParKalmanDefinitions.cuh"

std::vector<trackChecker::Tracks> prepareKalmanTracks(
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
  const char* scifi_geometry,
  const std::array<float, 9>& inv_clus_res,
  const ParKalmanFilter::FittedTrack* kf_tracks,
  const uint number_of_events);
