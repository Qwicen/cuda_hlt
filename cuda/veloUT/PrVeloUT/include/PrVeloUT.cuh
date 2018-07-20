#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include "../../../../main/include/Logger.h"
#include "../../../../main/include/SystemOfUnits.h"

#include "../../../../cuda/veloUT/common/include/VeloUTDefinitions.cuh"
#include "../../../../cuda/veloUT/PrVeloUT/include/PrVeloUTDefinitions.cuh"
#include "../../../../cuda/veloUT/PrVeloUT/include/PrVeloUTMagnetToolDefinitions.cuh"
#include "../../../../cuda/velo/common/include/VeloDefinitions.cuh"

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
  VeloState state;
  VeloUTTracking::Hit bestHits[VeloUTTracking::n_layers];
  int n_hits = 0;
  float bestParams[4];
  float wb, invKinkVeloDist, xMidField;

  __host__ __device__ TrackHelper(
    const VeloState& miniState
    ) : state(miniState) {
    bestParams[0] = bestParams[2] = bestParams[3] = 0.;
    bestParams[1] = PrVeloUTConst::maxPseudoChi2;
    xMidField = state.x + state.tx*(PrVeloUTConst::zKink-state.z);
    const float a = PrVeloUTConst::sigmaVeloSlope*(PrVeloUTConst::zKink - state.z);
    wb=1./(a*a);
    invKinkVeloDist = 1/(PrVeloUTConst::zKink-state.z);
    }
  };


__host__ __device__ bool veloTrackInUTAcceptance( const VeloState& state );

__host__ __device__ bool getHits(
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int posLayers[4][85],
  VeloUTTracking::HitsSoA *hits_layers,
  const float* fudgeFactors, 
  const VeloState& trState ); 

__host__ __device__ bool formClusters(
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int hitCandidateIndices[VeloUTTracking::n_layers],
  VeloUTTracking::HitsSoA *hits_layers,
  TrackHelper& helper,
  const bool forward);

__host__ __device__ void prepareOutputTrack(
  const uint* velo_track_hit_number,
  const VeloTracking::Hit<mc_check_enabled>* velo_track_hits,
  const int accumulated_tracks_event,
  const int i_track,
  const TrackHelper& helper,
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  VeloUTTracking::HitsSoA *hits_layers,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks],
  int* n_veloUT_tracks,
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
  VeloUTTracking::HitsSoA *hits_layers,
  int posLayers[4][85] );

__host__ __device__ void findHits( 
  const size_t posBeg,
  const size_t posEnd,
  VeloUTTracking::HitsSoA *hits_layers,
  const int layer_offset,
  const int i_layer,
  const VeloState& myState, 
  const float xTolNormFact,
  const float invNormFact,
  int hitCandidatesInLayer[VeloUTTracking::max_hit_candidates_per_layer],
  int &n_hitCandidatesInLayer,
  float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer]);

__host__ __device__ void addHits(
  float* mat,
  float* rhs,
  const VeloUTTracking::Hit** hits,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  VeloUTTracking::HitsSoA *hits_layers,
  const int n_hits);

__host__ __device__ void addChi2s(
  const float xUTFit,
  const float xSlopeUTFit,
  float& chi2,
  const VeloUTTracking::Hit** hits,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  VeloUTTracking::HitsSoA *hits_layers,
  const int n_hits );

template <int N>
__host__ __device__ void simpleFit(
  const VeloUTTracking::Hit* hits[N],
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  int bestHitCandidateIndices[VeloUTTracking::n_layers],
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  VeloUTTracking::HitsSoA *hits_layers,
  TrackHelper& helper ) {
  assert( N==3||N==4 );
  
#ifdef __CUDA_ARCH__
  // thrust::seq -> executed sequentially with one thread
  // thrust::device -> launch a child grid using
  // Dynamic Parallelism to parallelize the execution
  // the capture list of the lambda has to be captured by value ([=])
  // since a child kernel cannot access local memory of the parent kernel
  const int nHighThres = thrust::count_if(
                           thrust::seq, hits, hits + N,
                           [=]( const VeloUTTracking::Hit* hit ) {
                             return hit && hit->highThreshold(); });
#else
  const int nHighThres = std::count_if( 
                           hits,  hits + N,
                           []( const VeloUTTracking::Hit* hit ) {
                             return hit && hit->highThreshold(); });
#endif
  
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
  addHits( mat, rhs, hits, x_pos_layers, hitCandidateIndices, hitCandidatesInLayers, hits_layers, N );
// #ifndef __CUDACC__
//   std::for_each( hits, hits + N, [&](const VeloUTTracking::Hit* h) { addHit(mat,rhs,h); } );
// #else
//   thrust::for_each(
//     thrust::seq, hits, hits + N,
//     [=, &mat, &rhs](const VeloUTTracking::Hit* h) { addHit(mat,rhs,h); } );   
// #endif
  
  const float denom       = 1. / (mat[0]*mat[2] - mat[1]*mat[1]);
  const float xSlopeUTFit = 0.001*(mat[0]*rhs[1] - mat[1]*rhs[0]) * denom;
  const float xUTFit      = (mat[2]*rhs[0] - mat[1]*rhs[1]) * denom;
  
  // new VELO slope x
  const float xb = xUTFit+xSlopeUTFit*(PrVeloUTConst::zKink-PrVeloUTConst::zMidUT);
  const float xSlopeVeloFit = (xb-helper.state.x)*helper.invKinkVeloDist;
  const float chi2VeloSlope = (helper.state.tx - xSlopeVeloFit)*PrVeloUTConst::invSigmaVeloSlope;
  
  /* chi2 takes chi2 from velo fit + chi2 from UT fit */
  float chi2UT = chi2VeloSlope*chi2VeloSlope;
  addChi2s( xUTFit, xSlopeUTFit, chi2UT, hits, x_pos_layers, hitCandidateIndices, hitCandidatesInLayers, hits_layers, N );
// #ifndef __CUDACC__
//   std::for_each( hits, hits + N, [&](const VeloUTTracking::Hit* h) { addChi2(xUTFit,xSlopeUTFit, chi2UT, h); } );
// #else
//   thrust::for_each(
//     thrust::seq, hits, hits + N,
//     [=](const VeloUTTracking::Hit* h) { addChi2(xUTFit,xSlopeUTFit, chi2UT, h); } );
// #endif

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
      helper.bestHits[i_hit] = *(hits[i_hit]);
      bestHitCandidateIndices[i_hit] = hitCandidateIndices[i_hit];
    }
    helper.n_hits = N;

  }

}
 
  


