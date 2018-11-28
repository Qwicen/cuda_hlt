#pragma once

#include "VeloDefinitions.cuh"
#include "UTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "PrVeloUT.cuh"
#include "Handler.cuh"
#include "UTEventModel.cuh"

__global__ void veloUT(
  uint* dev_ut_hits,
  uint* dev_ut_hit_offsets,
  int* dev_atomics_velo,
  uint* dev_velo_track_hit_number,
  char* dev_velo_track_hits,
  char* dev_velo_states,
  UT::TrackHits* dev_veloUT_tracks,
  int* dev_atomics_ut,
  PrUTMagnetTool* dev_ut_magnet_tool,
  float* dev_ut_dxDy,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  const float* dev_unique_sector_xs
);

ALGORITHM(veloUT, veloUT_t)
