#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

#include "PrVeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "UTDefinitions.cuh"
#include "UTEventModel.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "MiniState.cuh"

/** PrVeloUT
 *
 *  @author Mariusz Witek
 *  @date   2007-05-08
 *  @update for A-Team framework 2007-08-20 SHM
 *
 *  2017-03-01: Christoph Hasse (adapt to future framework)
 *  2018-05-05: Plácido Fernández (make standalone)
 *  2018-07:    Dorothea vom Bruch (convert to C and then CUDA code)
 */
struct TrackHelper {
  int bestHitIndices[UT::Constants::n_layers] = {-1, -1, -1, -1};
  int n_hits = 0;
  float bestParams[4];
  float wb, invKinkVeloDist, xMidField;

  __host__ __device__ TrackHelper(const MiniState& state)
  {
    bestParams[0] = bestParams[2] = bestParams[3] = 0.;
    bestParams[1] = PrVeloUTConst::maxPseudoChi2;
    xMidField = state.x + state.tx * (PrVeloUTConst::zKink - state.z);
    const float a = PrVeloUTConst::sigmaVeloSlope * (PrVeloUTConst::zKink - state.z);
    wb = 1. / (a * a);
    invKinkVeloDist = 1 / (PrVeloUTConst::zKink - state.z);
  }
};

__host__ __device__ void propagate_state_to_end_velo(VeloState& velo_state);

__host__ __device__ bool veloTrackInUTAcceptance(const MiniState& state);

__device__ bool getHits(
  int hitCandidatesInLayers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[UT::Constants::n_layers],
  float x_pos_layers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  UT::Hits& ut_hits,
  UT::HitOffsets& ut_hit_offsets,
  const float* fudgeFactors,
  const MiniState& trState,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets);

__host__ __device__ bool formClusters(
  const int hitCandidatesInLayers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  const int n_hitCandidatesInLayers[UT::Constants::n_layers],
  const float x_pos_layers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  UT::Hits& ut_hits,
  UT::HitOffsets& ut_hit_offsets,
  TrackHelper& helper,
  MiniState& state,
  const float* ut_dxDy,
  const bool forward);

__host__ __device__ void prepareOutputTrack(
  const Velo::Consolidated::Hits& velo_track_hits,
  const uint velo_track_hit_number,
  const TrackHelper& helper,
  const MiniState& state,
  int hitCandidatesInLayers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[UT::Constants::n_layers],
  UT::Hits& ut_hits,
  UT::HitOffsets& ut_hit_offsets,
  const float x_pos_layers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  UT::TrackHits VeloUT_tracks[UT::Constants::max_num_tracks],
  int* n_veloUT_tracks,
  const int i_velo_track,
  const float* bdlTable,
  const int event_hit_offset);

__host__ __device__ void fillArray(int* array, const int size, const size_t value);

__host__ __device__ void fillArrayAt(int* array, const int offset, const int n_vals, const size_t value);

__host__ __device__ void fillIterators(UT::Hits& ut_hits, UT::HitOffsets& ut_hit_offsets, int posLayers[4][85]);

__host__ __device__ void findHits(
  const uint lowerBoundSectorGroup,
  const uint upperBoundSectorGroup,
  UT::Hits& ut_hits,
  UT::HitOffsets& ut_hit_offsets,
  uint layer_offset,
  const int i_layer,
  const float* ut_dxDy,
  const MiniState& myState,
  const float xTolNormFact,
  const float invNormFact,
  int hitCandidatesInLayer[UT::Constants::max_hit_candidates_per_layer],
  int& n_hitCandidatesInLayer,
  float x_pos_layers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer]);

// =================================================
// -- 2 helper functions for fit
// -- Pseudo chi2 fit, templated for 3 or 4 hits
// =================================================
template<int N, bool InOrder>
__host__ __device__ void addHits(
  float* mat,
  float* rhs,
  const float x_pos_layers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  const int hitCandidateIndices[UT::Constants::n_layers],
  const int hitCandidatesInLayers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  UT::Hits& ut_hits,
  const int hitIndices[N],
  const float* ut_dxDy,
  const bool forward)
{
  for (int i_hit = 0; i_hit < N; ++i_hit) {
    const int hit_index = hitIndices[i_hit];
    // const int planeCode = ut_hits.planeCode[hit_index];
    int planeCode = forward ? i_hit : 4 - 1 - i_hit;
    if (planeCode == 1 && !InOrder) {
      planeCode = 3;
    }
    else if (planeCode == 0 && !InOrder) {
      planeCode = 1;
    }
    const float ui = x_pos_layers[planeCode][hitCandidateIndices[i_hit]];
    const float dxDy = ut_dxDy[planeCode];
    const float ci = ut_hits.cosT(hit_index, dxDy);
    const float z = ut_hits.zAtYEq0[hit_index];
    const float dz = 0.001 * (z - PrVeloUTConst::zMidUT);
    const float wi = ut_hits.weight[hit_index];

    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }
}

template<int N, bool InOrder>
__host__ __device__ void addChi2s(
  const float xUTFit,
  const float xSlopeUTFit,
  float& chi2,
  const float x_pos_layers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  const int hitCandidateIndices[UT::Constants::n_layers],
  const int hitCandidatesInLayers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  UT::Hits& ut_hits,
  const int hitIndices[N],
  const bool forward)
{
  for (int i_hit = 0; i_hit < N; ++i_hit) {
    const int hit_index = hitIndices[i_hit];
    // const int planeCode = ut_hits.planeCode[hit_index];
    int planeCode = forward ? i_hit : 4 - 1 - i_hit;
    if (planeCode == 1 && !InOrder) {
      planeCode = 3;
    }
    else if (planeCode == 0 && !InOrder) {
      planeCode = 1;
    }
    const float zd = ut_hits.zAtYEq0[hit_index];
    const float xd = xUTFit + xSlopeUTFit * (zd - PrVeloUTConst::zMidUT);
    const float x = x_pos_layers[planeCode][hitCandidateIndices[i_hit]];
    const float du = xd - x;
    chi2 += (du * du) * ut_hits.weight[hit_index];
  }
}

template<int N, bool InOrder>
__host__ __device__ void simpleFit(
  const float x_pos_layers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  const int hitCandidateIndices[UT::Constants::n_layers],
  const int hitCandidatesInLayers[UT::Constants::n_layers][UT::Constants::max_hit_candidates_per_layer],
  UT::Hits& ut_hits,
  const int hitIndices[N],
  TrackHelper& helper,
  MiniState& state,
  const float* ut_dxDy,
  const bool forward)
{
  assert(N == 3 || N == 4);

  /* Straight line fit of UT hits,
     including the hit at x_mid_field, z_mid_field,
     use least squares method for fitting x(z) = a + bz,
     the chi2 is minimized and expressed in terms of sums as described
     in chapter 4 of http://cds.cern.ch/record/1635665/files/LHCb-PUB-2013-023.pdf
    */
  // -- Scale the z-component, to not run into numerical problems with floats
  // -- first add to sum values from hit at xMidField, zMidField hit
  const float zDiff = 0.001 * (PrVeloUTConst::zKink - PrVeloUTConst::zMidUT);
  float mat[3] = {helper.wb, helper.wb * zDiff, helper.wb * zDiff * zDiff};
  float rhs[2] = {helper.wb * helper.xMidField, helper.wb * helper.xMidField * zDiff};

  // then add to sum values from hits on track
  addHits<N, InOrder>(
    mat, rhs, x_pos_layers, hitCandidateIndices, hitCandidatesInLayers, ut_hits, hitIndices, ut_dxDy, forward);

  const float denom = 1. / (mat[0] * mat[2] - mat[1] * mat[1]);
  const float xSlopeUTFit = 0.001 * (mat[0] * rhs[1] - mat[1] * rhs[0]) * denom;
  const float xUTFit = (mat[2] * rhs[0] - mat[1] * rhs[1]) * denom;

  // new VELO slope x
  const float xb = xUTFit + xSlopeUTFit * (PrVeloUTConst::zKink - PrVeloUTConst::zMidUT);
  const float xSlopeVeloFit = (xb - state.x) * helper.invKinkVeloDist;
  const float chi2VeloSlope = (state.tx - xSlopeVeloFit) * PrVeloUTConst::invSigmaVeloSlope;

  /* chi2 takes chi2 from velo fit + chi2 from UT fit */
  float chi2UT = chi2VeloSlope * chi2VeloSlope;
  addChi2s<N, InOrder>(
    xUTFit,
    xSlopeUTFit,
    chi2UT,
    x_pos_layers,
    hitCandidateIndices,
    hitCandidatesInLayers,
    ut_hits,
    hitIndices,
    forward);

  chi2UT /= (N + 1 - 2);

  if (chi2UT < helper.bestParams[1]) {
    // calculate q/p
    const float sinInX = xSlopeVeloFit * std::sqrt(1. + xSlopeVeloFit * xSlopeVeloFit);
    const float sinOutX = xSlopeUTFit * std::sqrt(1. + xSlopeUTFit * xSlopeUTFit);
    const float qp = (sinInX - sinOutX);

    helper.bestParams[0] = qp;
    helper.bestParams[1] = chi2UT;
    helper.bestParams[2] = xUTFit;
    helper.bestParams[3] = xSlopeUTFit;
    helper.bestHitIndices[3] = -1;
    for (int i_hit = 0; i_hit < N; ++i_hit) {
      helper.bestHitIndices[i_hit] = hitIndices[i_hit];
    }

    helper.n_hits = N;
  }
}
