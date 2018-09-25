#pragma once

// #include "VeloDefinitions.cuh"
// #include "VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "PrVeloUT.cuh"

__global__ void compassUT(
  uint* dev_ut_hits, // actual hit content
  uint* dev_ut_hit_count, // prefixsum, count per layer
  int* dev_atomics_storage, // semi_prefixsum, offset to tracks
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  float* dev_ut_dxDy,
  int* dev_active_tracks,
  VeloUTTracking::TrackUT* dev_compassUT_tracks,
  int* dev_atomics_compassUT);