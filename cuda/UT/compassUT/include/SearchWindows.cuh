#pragma once

#include "VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"

__global__ void ut_search_windows(
  uint* dev_ut_hits,
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  const uint* dev_unique_x_sector_layer_offsets,
  const float* dev_unique_sector_xs,
  int* dev_windows_layers);
