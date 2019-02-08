#pragma once

#include "UTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "CompassUTDefinitions.cuh"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsVelo.cuh"
#include "ArgumentsUT.cuh"

__global__ void ut_search_windows(
  uint* dev_ut_hits,
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  char* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  const uint* dev_unique_x_sector_layer_offsets,
  const float* dev_unique_sector_xs,
  short* dev_windows_layers,
  int* dev_active_tracks,
  bool* dev_accepted_velo_tracks);

ALGORITHM(
  ut_search_windows,
  ut_search_windows_t,
  ARGUMENTS(
    dev_ut_hits,
    dev_ut_hit_offsets,
    dev_atomics_velo,
    dev_velo_track_hit_number,
    dev_velo_track_hits,
    dev_velo_states,
    dev_ut_windows_layers,
    dev_accepted_velo_tracks,
    dev_ut_active_tracks))
