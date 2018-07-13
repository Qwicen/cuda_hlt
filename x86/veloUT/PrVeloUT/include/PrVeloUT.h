#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

#include "../../../../main/include/Logger.h"
#include "../../../../main/include/SystemOfUnits.h"

#include "../../../../cuda/veloUT/common/include/VeloUTDefinitions.cuh"
#include "../../../../cuda/veloUT/PrVeloUT/include/PrVeloUTDefinitions.cuh"
#include "../../../../cuda/veloUT/PrVeloUT/include/PrVeloUTMagnetToolDefinitions.cuh"
#include "../../../../cuda/velo/common/include/VeloDefinitions.cuh"

/** @class PrVeloUT PrVeloUT.h
   *
   *  - InputTracksName: Input location for Velo tracks
   *  - OutputTracksName: Output location for VeloTT tracks
   *  - TimingMeasurement: Do a timing measurement?
   *
   *  @author Mariusz Witek
   *  @date   2007-05-08
   *  @update for A-Team framework 2007-08-20 SHM
   *
   *  2017-03-01: Christoph Hasse (adapt to future framework)
   *  2018-05-05: Plácido Fernández (make standalone)
   *  2018-07:    Dorothea vom Bruch (convert to C code for GPU compatability)
   */

struct TrackHelper{
  VeloState state;
  VeloUTTracking::Hit bestHits[VeloUTTracking::n_layers];
  int n_hits = 0;
  float bestParams[4];
  float wb, invKinkVeloDist, xMidField;

  TrackHelper(
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

class PrVeloUT {

public:
  
  void operator()(
    const uint* velo_track_hit_number,
    const VeloTracking::Hit<true>* velo_track_hits,
    const int number_of_tracks_event,
    const int accumulated_tracks_event,
    const VeloState* velo_states_event,
    VeloUTTracking::HitsSoA *hits_layers_events,
    const PrUTMagnetTool *magnet_tool,
    VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks],
    int &n_velo_tracks_in_UT,
    int &n_veloUT_tracks
  ) const;

private:

   bool veloTrackInUTAcceptance(
    const VeloState& state ) const;

  bool getHits(
    int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
    int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
    const int posLayers[4][85],
    VeloUTTracking::HitsSoA *hits_layers,
    const float* fudgeFactors, 
    const VeloState& trState ) const; 

  bool formClusters(
    const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
    const int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
    VeloUTTracking::HitsSoA *hits_layers,
    TrackHelper& helper,
    const bool forward ) const;

  void prepareOutputTrack(
    const uint* velo_track_hit_number,
    const VeloTracking::Hit<true>* velo_track_hits,
    const int accumulated_tracks_event,
    const int i_track,
    const TrackHelper& helper,
    int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
    int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
    VeloUTTracking::HitsSoA *hits_layers,
    VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks],
    int &n_veloUT_tracks,
    const float* bdlTable) const;

  void fillArray(
    int * array,
    const int size,
    const size_t value ) const {
    for ( int i = 0; i < size; ++i ) {
      array[i] = value;
    }
  }
  
  void fillArrayAt(
    int * array,
    const int offset,
    const int n_vals,
    const size_t value ) const {  
    fillArray( array + offset, n_vals, value ); 
  }
  
  // ==============================================================================
  // -- Method to cache some starting points for the search
  // -- This is actually faster than binary searching the full array
  // -- Granularity hardcoded for the moment.
  // -- Idea is: UTb has dimensions in x (at y = 0) of about -860mm -> 860mm
  // -- The indices go from 0 -> 84, and shift by -42, leading to -42 -> 42
  // -- Taking the higher density of hits in the center into account, the positions of the iterators are
  // -- calculated as index*index/2, where index = [ -42, 42 ], leading to
  // -- -882mm -> 882mm
  // -- The last element is an "end" iterator, to make sure we never go out of bound
  // ==============================================================================
  inline void fillIterators(
    VeloUTTracking::HitsSoA *hits_layers,
    int posLayers[4][85] ) const
  {
    
    for(int iStation = 0; iStation < 2; ++iStation){
      for(int iLayer = 0; iLayer < 2; ++iLayer){
        int layer = 2*iStation + iLayer;
       	int layer_offset = hits_layers->layer_offset[layer];
	
        size_t pos = 0;
        // to do: check whether there is an efficient thrust implementation for this
        fillArray( posLayers[layer], 85, pos );

        int bound = -42.0;
        // to do : make copysignf
        float val = std::copysign(float(bound*bound)/2.0, bound);

        // TODO add bounds checking
        for ( ; pos != hits_layers->n_hits_layers[layer]; ++pos) {
          while( hits_layers->xAtYEq0( layer_offset + pos ) > val){
            posLayers[layer][bound+42] = pos;
            ++bound;
            val = std::copysign(float(bound*bound)/2.0, bound);
          }
        }

        fillArrayAt(
          posLayers[layer],
          42 + bound,
          85 - 42 - bound,
          hits_layers->n_hits_layers[layer] );
        
      }
    }
  }

  // ==============================================================================
  // -- Finds the hits in a given layer within a certain range
  // ==============================================================================
  inline void findHits( 
    const size_t posBeg,
    const size_t posEnd,
    VeloUTTracking::HitsSoA *hits_layers,
    const int layer_offset,
    const VeloState& myState, 
    const float xTolNormFact,
    const float invNormFact,
    int hitCandidatesInLayer[VeloUTTracking::max_hit_candidates_per_layer],
    int &n_hitCandidatesInLayer
    ) const 
  {
    const auto zInit = hits_layers->zAtYEq0( layer_offset + posBeg );
    const auto yApprox = myState.y + myState.ty * (zInit - myState.z);

    // to do: use fabsf instead of std::abs
    size_t pos = posBeg;
    while ( 
      pos <= posEnd && 
      hits_layers->isNotYCompatible( layer_offset + pos, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(xTolNormFact) )
    ) { ++pos; }

    const auto xOnTrackProto = myState.x + myState.tx*(zInit - myState.z);
    const auto yyProto =       myState.y - myState.ty*myState.z;

    for (int i=pos; i<posEnd; ++i) {

      const auto xx = hits_layers->xAt( layer_offset + i, yApprox ); 
      const auto dx = xx - xOnTrackProto;
      
      if( dx < -xTolNormFact ) continue;
      if( dx >  xTolNormFact ) break; 

      // -- Now refine the tolerance in Y
      if ( hits_layers->isNotYCompatible( layer_offset + i, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx*invNormFact)) ) continue;
      
      
      const auto zz = hits_layers->zAtYEq0( layer_offset + i ); 
      const auto yy = yyProto +  myState.ty*zz;
      const auto xx2 = hits_layers->xAt( layer_offset + i, yy );

      hits_layers->x[ layer_offset + i ] = xx2;
      hits_layers->z[ layer_offset + i ] = zz;

      hitCandidatesInLayer[n_hitCandidatesInLayer] = i;
      n_hitCandidatesInLayer++;
      
      if ( n_hitCandidatesInLayer >= VeloUTTracking::max_hit_candidates_per_layer )
        debug_cout << "n hits candidates = " << n_hitCandidatesInLayer << std::endl;
      assert( n_hitCandidatesInLayer < VeloUTTracking::max_hit_candidates_per_layer );
    }
    for ( int i_hit = 0; i_hit < n_hitCandidatesInLayer; ++i_hit ) {
      if ( hitCandidatesInLayer[i_hit] >= VeloUTTracking::max_numhits_per_event )
        debug_cout << "hit index = " << hitCandidatesInLayer[i_hit] << std::endl;
      assert( hitCandidatesInLayer[i_hit] < VeloUTTracking::max_numhits_per_event );
    }
  }

  // ===========================================================================================
  // -- 2 helper functions for fit
  // -- Pseudo chi2 fit, templated for 3 or 4 hits
  // ===========================================================================================
  void addHit( float* mat, float* rhs, const VeloUTTracking::Hit* hit) const {
    const float ui = hit->x;
    const float ci = hit->cosT();
    const float dz = 0.001*(hit->z - PrVeloUTConst::zMidUT);
    const float wi = hit->weight();

    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }

  void addChi2( const float xTTFit, const float xSlopeTTFit, float& chi2 , const VeloUTTracking::Hit* hit) const {
    const float zd    = hit->z;
    const float xd    = xTTFit + xSlopeTTFit*(zd-PrVeloUTConst::zMidUT);
    const float du    = xd - hit->x;
    chi2 += (du*du)*hit->weight();
  }

  template <int N>
  void simpleFit(
    const VeloUTTracking::Hit* hits[N], 
    TrackHelper& helper ) const 
  {
    assert( N==3||N==4 );

    // to do in cuda: implement count_if / use thrust
    const int nHighThres = std::count_if( 
                                         hits,  hits + N, []( const VeloUTTracking::Hit* hit ) { return hit && hit->highThreshold(); });
    
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

    // to do in cuda: implement for_each / use thrust
    // then add to sum values from hits on track
    std::for_each( hits, hits + N, [&](const VeloUTTracking::Hit* h) { this->addHit(mat,rhs,h); } );

    const float denom       = 1. / (mat[0]*mat[2] - mat[1]*mat[1]);
    const float xSlopeUTFit = 0.001*(mat[0]*rhs[1] - mat[1]*rhs[0]) * denom;
    const float xUTFit      = (mat[2]*rhs[0] - mat[1]*rhs[1]) * denom;

    // new VELO slope x
    const float xb = xUTFit+xSlopeUTFit*(PrVeloUTConst::zKink-PrVeloUTConst::zMidUT);
    const float xSlopeVeloFit = (xb-helper.state.x)*helper.invKinkVeloDist;
    const float chi2VeloSlope = (helper.state.tx - xSlopeVeloFit)*PrVeloUTConst::invSigmaVeloSlope;

    /* chi2 takes chi2 from velo fit + chi2 from UT fit */
    float chi2UT = chi2VeloSlope*chi2VeloSlope;
    // to do: use thrust call
    std::for_each( hits, hits + N, [&](const VeloUTTracking::Hit* h) { this->addChi2(xUTFit,xSlopeUTFit, chi2UT, h); } );

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
      }
      helper.n_hits = N;
    }

  }

  
};

