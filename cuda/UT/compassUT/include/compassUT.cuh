#pragma once

#include "PrVeloUT.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "VeloDefinitions.cuh"
#include "VeloUTDefinitions.cuh"

constexpr uint N_LAYERS = VeloUTTracking::n_layers;

//=========================================================================
// Point to correct position for windows pointers
//=========================================================================
struct LayerCandidates {
  int from0;
  int to0;
  int from1;
  int to1;
  int from2;
  int to2;
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
  float xUTFit; // TODO check we need this
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
  uint* dev_ut_hits,
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  int* dev_active_tracks,
  const uint* dev_unique_x_sector_layer_offsets,
  const float* dev_unique_sector_xs,
  VeloUTTracking::TrackUT* dev_compassUT_tracks,
  int* dev_atomics_compassUT,
  int* dev_windows_layers);

__host__ __device__ bool velo_track_in_UT_acceptance(
  const MiniState& state);

__host__ __device__ bool check_tol_refine(
  const int hit_index,
  const UTHits& ut_hits,
  const MiniState& velo_state,
  const float normFactNum,
  const float xTol,
  const float dxDy);

__host__ __device__ void find_best_hits(
  const int i_track,
  const int* dev_windows_layers,
  const std::tuple<int, int, int, int, int, int>* candidates_layers,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const MiniState& velo_state,
  const float* fudgeFactors,
  const float* ut_dxDy,
  const bool forward,
  int* best_hits,
  BestParams& best_params);

__host__ __device__ BestParams pkick_fit(
  const int best_hits[N_LAYERS],
  const UTHits& ut_hits,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const float yyProto);

__device__ void save_track(
  const int i_track,
  const float* bdlTable,
  const MiniState& velo_state,
  const BestParams& best_params,
  uint* dev_velo_track_hits,
  const Velo::Consolidated::Tracks& velo_tracks,
  const int num_best_hits,
  const int* best_hits,
  const UTHits& ut_hits,
  const float* ut_dxDy,
  int* n_veloUT_tracks,
  VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks]);