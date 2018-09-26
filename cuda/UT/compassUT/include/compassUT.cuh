#pragma once

#include "VeloDefinitions.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "PrVeloUT.cuh"

__global__ void compassUT(
  uint* dev_ut_hits, // actual hit content
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage, // semi_prefixsum, offset to tracks
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  int* dev_active_tracks,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  const float* dev_unique_sector_xs,
  VeloUTTracking::TrackUT* dev_compassUT_tracks,
  int* dev_atomics_compassUT, // size of number of events
  int* dev_windows_layers);