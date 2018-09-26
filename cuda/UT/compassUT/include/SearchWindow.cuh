#pragma once

#include "UTDefinitions.cuh"
#include "VeloConsolidated.cuh"

// #include "SystemOfUnits.h"
// #include "VeloEventModel.cuh"
// #include "VeloUTDefinitions.cuh"
// #include "PrVeloUTDefinitions.cuh"
// #include "PrVeloUTMagnetToolDefinitions.h"
// #include "UTDefinitions.cuh"

constexpr int N_LAYERS = 4;

struct BasicState {
  float x, y, tx, ty, z;

  __device__ BasicState(
    const Velo::Consolidated::States& velo_states,
    const uint index
  ) : x(velo_states.x[index]),
    y(velo_states.y[index]),
    tx(velo_states.tx[index]),
    ty(velo_states.ty[index]),
    z(velo_states.z[index]) {}
};

__host__ __device__ bool velo_track_in_UTA_acceptance(
  const BasicState& state
);

__device__ void binary_search_range(
  const int layer,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_offsets,
  const float ut_dxDy,
  const float low_bound_x,
  const float up_bound_x,
  const float xTolNormFact,
  const float yApprox,
  const float xOnTrackProto,
  const int layer_offset,
  const uint first_sector_group_in_layer,
  const uint last_sector_group_in_layer,
  const float* dev_unique_sector_xs,
  int& high_hit_pos,
  int& low_hit_pos);

__device__ void get_windows(
  const int i_track,
  const BasicState& veloState,
  const float* fudgeFactors,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets,
  const Velo::Consolidated::Tracks& velo_tracks,
  int* windows_layers);