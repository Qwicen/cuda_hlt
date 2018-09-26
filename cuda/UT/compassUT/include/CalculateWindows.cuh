#pragma once

#include "UTDefinitions.cuh"
#include "VeloConsolidated.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"

__device__ bool velo_track_in_UTA_acceptance(const MiniState& state);

__device__ std::tuple<int, int> calculate_windows(
  const int i_track,
  const int layer,
  const MiniState& velo_state,
  const float* fudge_factors,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets,
  const Velo::Consolidated::Tracks& velo_tracks);