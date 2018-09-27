#pragma once

#include "PrVeloUT.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "VeloDefinitions.cuh"
#include "VeloUTDefinitions.cuh"

constexpr uint N_LAYERS = 4;

struct layer_candidates {
  int first;
  int last;
};

struct track_candidates {
  layer_candidates layer[N_LAYERS];
};

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

__host__ __device__ bool find_best_hits(
  const int i_track,
  const int* windows_layers,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const bool forward,
  TrackHelper& helper,
  float* x_hit_layer,
  int* bestHitCandidateIndices);

// =================================================
// -- 2 helper functions for fit
// -- Pseudo chi2 fit, templated for 3 or 4 hits
// =================================================
template<int N>
__host__ __device__ void add_hits(
  float* mat,
  float* rhs,
  const float* x_hit_layer,
  const UTHits& ut_hits,
  const int hitIndices[N],
  const float* ut_dxDy)
{
  for (int i_hit = 0; i_hit < N; ++i_hit) {
    const int hit_index = hitIndices[i_hit];
    const int planeCode = ut_hits.planeCode[hit_index];
    const float ui      = x_hit_layer[planeCode];
    const float dxDy    = ut_dxDy[planeCode];
    const float ci      = ut_hits.cosT(hit_index, dxDy);
    const float z       = ut_hits.zAtYEq0[hit_index];
    const float dz      = 0.001 * (z - PrVeloUTConst::zMidUT);
    const float wi      = ut_hits.weight[hit_index];

    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }
}

template<int N>
__host__ __device__ void add_chi2s(
  float& chi2,
  const float xUTFit,
  const float xSlopeUTFit,
  const float* x_hit_layer,
  const UTHits& ut_hits,
  const int hitIndices[N])
{
  for (int i_hit = 0; i_hit < N; ++i_hit) {
    const int hit_index = hitIndices[i_hit];
    const int planeCode = ut_hits.planeCode[hit_index];
    const float zd      = ut_hits.zAtYEq0[hit_index];
    const float xd      = xUTFit + xSlopeUTFit * (zd - PrVeloUTConst::zMidUT);
    const float x       = x_hit_layer[planeCode];
    const float du      = xd - x;
    chi2 += (du * du) * ut_hits.weight[hit_index];
  }
}

template<int N>
__host__ __device__ int count_hits_high_threshold(const int hitIndices[N], const UTHits& ut_hits)
{
  int nHighThres = 0;
  for (int i_hit = 0; i_hit < N; ++i_hit) { nHighThres += ut_hits.highThreshold[hitIndices[i_hit]]; }
  return nHighThres;
}

template<int N>
__host__ __device__ void simple_fit(
  const float* x_hit_layer,
  const int hitCandidateIndices[N_LAYERS],
  const UTHits& ut_hits,
  const int hitIndices[N],
  const MiniState& velo_state,
  const float* ut_dxDy,
  int bestHitCandidateIndices[N_LAYERS],
  TrackHelper& helper)
{
  assert(N == 3 || N == 4);

  const int nHighThres = count_hits_high_threshold<N>(hitIndices, ut_hits);

  // -- Veto hit combinations with no high threshold hit
  // -- = likely spillover
  if (nHighThres < PrVeloUTConst::minHighThres) return;

  // Straight line fit of UT hits,
  // including the hit at x_mid_field, z_mid_field,
  // use least squares method for fitting x(z) = a + bz,
  // the chi2 is minimized and expressed in terms of sums as described
  // in chapter 4 of http://cds.cern.ch/record/1635665/files/LHCb-PUB-2013-023.pdf

  // -- Scale the z-component, to not run into numerical problems with floats
  // -- first add to sum values from hit at xMidField, zMidField hit
  const float zDiff = 0.001 * (PrVeloUTConst::zKink - PrVeloUTConst::zMidUT);
  float mat[3]      = {helper.wb, helper.wb * zDiff, helper.wb * zDiff * zDiff};
  float rhs[2]      = {helper.wb * helper.xMidField, helper.wb * helper.xMidField * zDiff};

  // then add to sum values from hits on track
  add_hits<N>(mat, rhs, x_hit_layer, ut_hits, hitIndices, ut_dxDy);

  const float denom       = 1. / (mat[0] * mat[2] - mat[1] * mat[1]);
  const float xSlopeUTFit = 0.001 * (mat[0] * rhs[1] - mat[1] * rhs[0]) * denom;
  const float xUTFit      = (mat[2] * rhs[0] - mat[1] * rhs[1]) * denom;

  // new VELO slope x
  const float xb            = xUTFit + xSlopeUTFit * (PrVeloUTConst::zKink - PrVeloUTConst::zMidUT);
  const float xSlopeVeloFit = (xb - velo_state.x) * helper.invKinkVeloDist;
  const float chi2VeloSlope = (velo_state.tx - xSlopeVeloFit) * PrVeloUTConst::invSigmaVeloSlope;

  /* chi2 takes chi2 from velo fit + chi2 from UT fit */
  float chi2UT = chi2VeloSlope * chi2VeloSlope;
  add_chi2s<N>(chi2UT, xUTFit, xSlopeUTFit, x_hit_layer, ut_hits, hitIndices);

  chi2UT /= (N + 1 - 2);

  if (chi2UT < helper.bestParams[1]) {
    // calculate q/p
    const float sinInX  = xSlopeVeloFit * std::sqrt(1. + xSlopeVeloFit * xSlopeVeloFit);
    const float sinOutX = xSlopeUTFit * std::sqrt(1. + xSlopeUTFit * xSlopeUTFit);
    const float qp      = (sinInX - sinOutX);

    helper.bestParams[0] = qp;
    helper.bestParams[1] = chi2UT;
    helper.bestParams[2] = xUTFit;
    helper.bestParams[3] = xSlopeUTFit;

    // Copy the selected hits to the helper
    for (int i_hit = 0; i_hit < N; ++i_hit) {
      helper.bestHitIndices[i_hit]   = hitIndices[i_hit];
      bestHitCandidateIndices[i_hit] = hitCandidateIndices[i_hit];
    }
    helper.n_hits = N;
  }
}