#pragma once

#include "PrVeloUT.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "VeloUTDefinitions.cuh"
#include "CompassUTDefinitions.cuh"
#include "FindBestHits.cuh"

__device__ void find_best_hits(
  const int* win_size_shared,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const MiniState& velo_state,
  const float* fudgeFactors,
  const float* ut_dxDy,
  const bool forward,
  int* best_hits,
  BestParams& best_params);

__device__ BestParams pkick_fit(
  const int best_hits[N_LAYERS],
  const UTHits& ut_hits,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const float yyProto,
  const bool forward);

__device__ __inline__ int set_index(
  const int i, 
  const LayerCandidates& layer_cand);

__host__ __device__ __inline__ bool check_tol_refine(
  const int hit_index,
  const UTHits& ut_hits,
  const MiniState& velo_state,
  const float normFactNum,
  const float xTol,
  const float dxDy);