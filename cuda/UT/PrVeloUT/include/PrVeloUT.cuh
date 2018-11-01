#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cassert>

#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrVeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "UTDefinitions.cuh"
#include "VeloConsolidated.cuh"
#include "VeloEventModel.cuh"

struct MiniState {
  float x, y, tx, ty, z;

  __host__ __device__ MiniState(
    const Velo::Consolidated::States& velo_states,
    const uint index
  ) : x(velo_states.x[index]),
    y(velo_states.y[index]),
    tx(velo_states.tx[index]),
    ty(velo_states.ty[index]),
    z(velo_states.z[index]) {}
};

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
struct TrackHelper{
  int bestHitIndices[VeloUTTracking::n_layers];
  int n_hits = 0;
  float bestParams[4];
  float wb, invKinkVeloDist, xMidField;

  __host__ __device__ TrackHelper(
    const MiniState& state
  ) {
    bestParams[0] = bestParams[2] = bestParams[3] = 0.;
    bestParams[1] = PrVeloUTConst::maxPseudoChi2;
    xMidField = state.x + state.tx*(PrVeloUTConst::zKink-state.z);
    const float a = PrVeloUTConst::sigmaVeloSlope*(PrVeloUTConst::zKink - state.z);
    wb=1./(a*a);
    invKinkVeloDist = 1/(PrVeloUTConst::zKink-state.z);
    }
  };

__host__ __device__ void propagate_state_to_end_velo( Velo::State& velo_state );

__host__ __device__ bool veloTrackInUTAcceptance(
  const MiniState& state
);

__device__ bool getHits(
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  UTHits& ut_hits,
  UTHitOffsets& ut_hit_offsets,
  const float* fudgeFactors, 
  const MiniState& trState,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets); 

__host__ __device__ bool formClusters(
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int hitCandidateIndices[VeloUTTracking::n_layers],
  UTHits& ut_hits,
  UTHitOffsets& ut_hit_offsets,
  TrackHelper& helper,
  MiniState& state,
  const float* ut_dxDy,
  const bool forward);

__host__ __device__ void prepareOutputTrack(
  const Velo::Consolidated::Hits& velo_track_hits,
  const uint velo_track_hit_number,
  const TrackHelper& helper,
  const MiniState& state,
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  UTHits& ut_hits,
  UTHitOffsets& ut_hit_offsets,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks],
  int* n_veloUT_tracks,
  const int i_velo_track,
  const float* bdlTable);

__host__ __device__ void fillArray(
  int * array,
  const int size,
  const size_t value);
  
__host__ __device__ void fillArrayAt(
  int * array,
  const int offset,
  const int n_vals,
  const size_t value );
  
__host__ __device__ void fillIterators(
  UTHits& ut_hits,
  UTHitOffsets& ut_hit_offsets,
  int posLayers[4][85] );

__host__ __device__ void findHits( 
  const uint lowerBoundSectorGroup,
  const uint upperBoundSectorGroup,
  UTHits& ut_hits,
  UTHitOffsets& ut_hit_offsets,
  uint layer_offset,
  const int i_layer,
  const float* ut_dxDy,
  const MiniState& myState, 
  const float xTolNormFact,
  const float invNormFact,
  int hitCandidatesInLayer[VeloUTTracking::max_hit_candidates_per_layer],
  int &n_hitCandidatesInLayer,
  float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer]);


// =================================================
// -- 2 helper functions for fit
// -- Pseudo chi2 fit, templated for 3 or 4 hits
// =================================================
template <int N>
__host__ __device__ void addHits(
  float* mat,
  float* rhs,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  UTHits& ut_hits,
  const int hitIndices[N],
  const float* ut_dxDy
) {
  
  for ( int i_hit = 0; i_hit < N; ++i_hit ) {
    const int hit_index = hitIndices[i_hit];
    const int planeCode = ut_hits.planeCode[hit_index];
    const float ui = x_pos_layers[ planeCode ][ hitCandidateIndices[i_hit] ];
    const float dxDy = ut_dxDy[planeCode];
    const float ci = ut_hits.cosT(hit_index, dxDy);
    const float z  = ut_hits.zAtYEq0[hit_index];
    const float dz = 0.001*(z - PrVeloUTConst::zMidUT);
    const float wi = ut_hits.weight[hit_index];

    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }
}

template <int N>
__host__ __device__ void addChi2s(
  const float xUTFit,
  const float xSlopeUTFit,
  float& chi2 ,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  UTHits& ut_hits,
  const int hitIndices[N] ) {
  
  for ( int i_hit = 0; i_hit < N; ++i_hit ) {
    const int hit_index = hitIndices[i_hit];
    const int planeCode = ut_hits.planeCode[hit_index];
    const float zd = ut_hits.zAtYEq0[hit_index];
    const float xd = xUTFit + xSlopeUTFit*(zd-PrVeloUTConst::zMidUT);
    const float x  = x_pos_layers[ planeCode ][ hitCandidateIndices[i_hit] ];
    const float du = xd - x;
    chi2 += (du*du)*ut_hits.weight[hit_index];
  }
}


template <int N>
__host__ __device__ int countHitsWithHighThreshold(
  const int hitIndices[N],
  UTHits& ut_hits ) {

  int nHighThres = 0;
  for ( int i_hit = 0; i_hit < N; ++i_hit ) {
    nHighThres += ut_hits.highThreshold[hitIndices[i_hit]];
  }
  return nHighThres;
}

template <int N>
__host__ __device__ void simpleFit(
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  int bestHitCandidateIndices[VeloUTTracking::n_layers],
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  UTHits& ut_hits,
  const int hitIndices[N],
  TrackHelper& helper,
  MiniState& state,
  const float* ut_dxDy) {
  assert( N==3||N==4 );
 
  const int nHighThres = countHitsWithHighThreshold<N>(hitIndices, ut_hits);
  
  // -- Veto hit combinations with no high threshold hit
  // -- = likely spillover
  if( nHighThres < PrVeloUTConst::minHighThres ) return;
  
  /* Straight line fit of UT hits,
     including the hit at x_mid_field, z_mid_field,
     use least squares method for fitting x(z) = a + bz,
     the chi2 is minimized and expressed in terms of sums as described
     in chapter 4 of http://cds.cern.ch/record/1635665/files/LHCb-PUB-2013-023.pdf
    */
  // -- Scale the z-component, to not run into numerical problems with floats
  // -- first add to sum values from hit at xMidField, zMidField hit
  const float zDiff = 0.001*(PrVeloUTConst::zKink-PrVeloUTConst::zMidUT);
  float mat[3] = { helper.wb, helper.wb*zDiff, helper.wb*zDiff*zDiff };
  float rhs[2] = { helper.wb* helper.xMidField, helper.wb*helper.xMidField*zDiff };
  
  // then add to sum values from hits on track
  addHits<N>( mat, rhs, x_pos_layers, hitCandidateIndices, hitCandidatesInLayers, ut_hits, hitIndices, ut_dxDy );
  
  const float denom       = 1. / (mat[0]*mat[2] - mat[1]*mat[1]);
  const float xSlopeUTFit = 0.001*(mat[0]*rhs[1] - mat[1]*rhs[0]) * denom;
  const float xUTFit      = (mat[2]*rhs[0] - mat[1]*rhs[1]) * denom;
  
  // new VELO slope x
  const float xb = xUTFit+xSlopeUTFit*(PrVeloUTConst::zKink-PrVeloUTConst::zMidUT);
  const float xSlopeVeloFit = (xb-state.x)*helper.invKinkVeloDist;
  const float chi2VeloSlope = (state.tx - xSlopeVeloFit)*PrVeloUTConst::invSigmaVeloSlope;
  
  /* chi2 takes chi2 from velo fit + chi2 from UT fit */
  float chi2UT = chi2VeloSlope*chi2VeloSlope;
  addChi2s<N>( xUTFit, xSlopeUTFit, chi2UT, x_pos_layers, hitCandidateIndices, hitCandidatesInLayers, ut_hits, hitIndices );

  chi2UT /= (N + 1 - 2);

  if( chi2UT < helper.bestParams[1] ){
    // calculate q/p
    const float sinInX  = xSlopeVeloFit * std::sqrt(1.+xSlopeVeloFit*xSlopeVeloFit);
    const float sinOutX = xSlopeUTFit * std::sqrt(1.+xSlopeUTFit*xSlopeUTFit);
    const float qp = (sinInX-sinOutX);
    
    helper.bestParams[0] = qp;
    helper.bestParams[1] = chi2UT;
    helper.bestParams[2] = xUTFit;
    helper.bestParams[3] = xSlopeUTFit;
    
    for ( int i_hit = 0; i_hit < N; ++i_hit ) {
      helper.bestHitIndices[i_hit] = hitIndices[i_hit];
      bestHitCandidateIndices[i_hit] = hitCandidateIndices[i_hit];
    }
    helper.n_hits = N;
  }
}
