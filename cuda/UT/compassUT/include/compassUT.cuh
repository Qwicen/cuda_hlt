#pragma once

#include "PrVeloUT.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "VeloDefinitions.cuh"
#include "VeloUTDefinitions.cuh"

constexpr uint N_LAYERS = 4;

//=========================================================================
// Point to correct position for windows pointers
//=========================================================================
struct LayerCandidates {
  uint first;
  uint last;
};

struct TrackCandidates {
  LayerCandidates layer[N_LAYERS];
};

struct WindowIndicator {
  const int* m_base_pointer;
  __host__ __device__ WindowIndicator(const int* base_pointer) : m_base_pointer(base_pointer) {}

  __host__ __device__ const TrackCandidates* get_track_candidates(const int i_track)
  {
    return reinterpret_cast<const TrackCandidates*>(m_base_pointer + (2 * N_LAYERS * i_track));
  }
};

//=========================================================================
// Save the best parameters
//=========================================================================
struct BestParams {
  float qp;
  float chi2UT;
  float xUTFit;
  float xSlopeUTFit;

  __host__ __device__ BestParams () 
  {
    qp = 0.;
    chi2UT = PrVeloUTConst::maxPseudoChi2;
    xUTFit = 0.;
    xSlopeUTFit = 0.;
  }
};

//=========================================================================
// Functions definitions
//=========================================================================
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

__host__ __device__ std::tuple<int, int, int, int> find_best_hits(
  const int i_track,
  const int* windows_layers,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const bool forward,
  TrackHelper& helper,
  float* x_hit_layer);
// int* bestHitCandidateIndices);

__host__ __device__ BestParams pkick_fit(
  // const std::tuple<int,int,int,int>& best_hits,
  const int best_hits[N_LAYERS],
  const UTHits& ut_hits,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const float yyProto);