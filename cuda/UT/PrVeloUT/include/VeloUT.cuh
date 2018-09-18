#pragma once

#include "VeloDefinitions.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "PrVeloUT.cuh"

__global__ void veloUT(
  uint* dev_ut_hits,
  uint* dev_ut_hit_count,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  VeloUTTracking::TrackUT* dev_veloUT_tracks,
  int* dev_atomics_veloUT,
  PrUTMagnetTool* dev_ut_magnet_tool,
  float* dev_ut_dxDy,
  int* dev_active_tracks);

// vertical processing
__device__ bool process_track(
  const int i_track,
  const uint event_tracks_offset,
  const Velo::Consolidated::States& velo_states,
  int (&hitCandidatesInLayers)[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int (&n_hitCandidatesInLayers)[VeloUTTracking::n_layers],
  float (&x_pos_layers)[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int (&posLayers)[VeloUTTracking::n_layers][VeloUTTracking::n_iterations_pos],
  UTHits& ut_hits,
  UTHitCount& ut_hit_count,
  const float* fudgeFactors,
  float* dev_ut_dxDy);

// horizontal processing
__device__ void process_track2 (
  const int i_track,
  const uint event_tracks_offset,
  const Velo::Consolidated::States& velo_states,
  int (&hitCandidatesInLayers)[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int (&n_hitCandidatesInLayers)[VeloUTTracking::n_layers],
  float (&x_pos_layers)[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  UTHits& ut_hits,
  UTHitCount& ut_hit_count,
  uint* dev_velo_track_hits,
  const Velo::Consolidated::Tracks& velo_tracks,
  int* n_veloUT_tracks_event,
  VeloUTTracking::TrackUT* veloUT_tracks_event,
  const float* bdlTable,
  float* dev_ut_dxDy);