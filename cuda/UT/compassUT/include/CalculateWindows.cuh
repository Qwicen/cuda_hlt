#pragma once

#include "UTDefinitions.cuh"
#include "VeloConsolidated.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "UTEventModel.cuh"
#include "MiniState.cuh"
#include <tuple>

__device__ bool velo_track_in_UTA_acceptance(const MiniState& state);

__device__ std::tuple<int, int, int, int, int, int, int, int, int, int> calculate_windows(
  const int i_track,
  const int layer,
  const MiniState& velo_state,
  const float* fudge_factors,
  const UT::Hits& ut_hits,
  const UT::HitOffsets& ut_hit_count,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets,
  const Velo::Consolidated::Tracks& velo_tracks);

__device__ std::tuple<int, int> find_candidates_in_sector_group(
  const UT::Hits& ut_hits,
  const UT::HitOffsets& ut_hit_offsets,
  const MiniState& velo_state,
  const float* dev_unique_sector_xs,
  const float x_track,
  const float y_track,
  const float dx_dy,
  const float normFact,
  const float invNormFact,
  const float xTol,
  const int sector_group);

__host__ __device__ void tol_refine (
  int& first_candidate,
  int& last_candidate,
  const UT::Hits& ut_hits,
  const MiniState& velo_state,
  const float invNormfact,
  const float xTolNormFact,
  const float dxDy);
