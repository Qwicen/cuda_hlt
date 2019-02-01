#pragma once

#include "PrVeloUT.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "UTDefinitions.cuh"
#include "CompassUTDefinitions.cuh"
#include "FindBestHits.cuh"
#include <tuple>

__device__ std::tuple<int,int,int,int,BestParams> find_best_hits(
  const short* win_size_shared,
  const uint number_of_tracks_event,
  const int i_track,
  const UT::Hits& ut_hits,
  const UT::HitOffsets& ut_hit_offsets,
  const MiniState& velo_state,
  const float* ut_dxDy);

__device__ BestParams pkick_fit(
  const int best_hits[UT::Constants::n_layers],
  const UT::Hits& ut_hits,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const float yyProto,
  const bool forward);

__device__ __inline__ int sum_layer_hits(
  const TrackCandidates& ranges, const int layer0, const int layer2);

__device__ __inline__ int sum_layer_hits(
  const TrackCandidates& ranges, const int layer);

__device__ __inline__ int calc_index(
  const int i, 
  const TrackCandidates& ranges, 
  const int layer,
  const UT::HitOffsets& ut_hit_offsets);

__device__ __inline__ int calc_index(
  const int i, 
  const TrackCandidates& ranges, 
  const int layer0,
  const int layer2,
  const UT::HitOffsets& ut_hit_offsets);

__device__ __inline__ bool check_tol_refine(
  const int hit_index,
  const UT::Hits& ut_hits,
  const MiniState& velo_state,
  const float normFactNum,
  const float xTol,
  const float dxDy);