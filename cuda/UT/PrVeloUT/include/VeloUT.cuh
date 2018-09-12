#pragma once

#include "VeloDefinitions.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "PrVeloUT.cuh"

__global__ void veloUT(
  uint* dev_ut_hits,
  uint* dev_ut_hit_count,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  const VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  const VeloState* dev_velo_states,
  VeloUTTracking::TrackUT* dev_veloUT_tracks,
  int* dev_atomics_veloUT,
  const PrUTMagnetTool* dev_ut_magnet_tool,
  float* dev_ut_dxDy
);
