#pragma once

#include "../../../velo/common/include/VeloDefinitions.cuh"
#include "../../common/include/VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.cuh"

__global__ void veloUT(
  VeloUTTracking::HitsSoA* dev_ut_hits,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states,
  VeloUTTracking::TrackUT *dev_veloUT_tracks,
  PrUTMagnetTool* dev_ut_magnet_tool
);
