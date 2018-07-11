#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

// #include "../../PrUTMagnetTool/PrUTMagnetTool.h"
#include "../../main/include/Logger.h"

// Math from ROOT
#include "../include/CholeskyDecomp.h"

#include "../include/SystemOfUnits.h"

#include "../../cuda/veloUT/common/include/VeloUTDefinitions.cuh"
#include "../../cuda/velo/common/include/VeloDefinitions.cuh"

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
   */

struct PrUTMagnetTool {
  static const int N_dxLay_vals = 124;
  static const int N_bdl_vals   = 3752;
  
  //const float m_zMidUT = 0.0;
  //const float m_averageDist2mom = 0.0;
  //std::vector<float> dxLayTable;
  float* dxLayTable;
  //std::vector<float> bdlTable;
  float* bdlTable;

  PrUTMagnetTool(){}
  PrUTMagnetTool( 
    const float *_dxLayTable, 
    const float *_bdlTable ) {
    dxLayTable = new float[N_dxLay_vals];
    for ( int i = 0; i < N_dxLay_vals; ++i ) {
      dxLayTable[i] = _dxLayTable[i];
    }
    bdlTable = new float[N_bdl_vals];
    for ( int i = 0; i < N_bdl_vals; ++i ) {
      bdlTable[i] = _bdlTable[i];
    }
  }
  
  //float zMidUT() { return m_zMidUT; }
  //float averageDist2mom() { return m_averageDist2mom; }
  float* returnDxLayTable() const { return dxLayTable; }
  float* returnBdlTable() const { return bdlTable; }
};

struct TrackHelper{
  VeloState state;
  VeloUTTracking::Hit bestHits[VeloUTTracking::n_layers];
  int n_hits = 0;
  float bestParams[4];
  float wb, invKinkVeloDist, xMidField;

  TrackHelper(
    const VeloState& miniState, 
    const float zKink, 
    const float sigmaVeloSlope, 
    const float maxPseudoChi2
    ) : state(miniState) {
    bestParams[0] = bestParams[2] = bestParams[3] = 0.;
    bestParams[1] = maxPseudoChi2;
    xMidField = state.x + state.tx*(zKink-state.z);
    const float a = sigmaVeloSlope*(zKink - state.z);
    wb=1./(a*a);
    invKinkVeloDist = 1/(zKink-state.z);
    }
  };

class PrVeloUT {

public:

  virtual int initialize();
  
  std::vector<VeloUTTracking::TrackUT> operator()(
    const uint* velo_track_hit_number,
    const VeloTracking::Hit<true>* velo_track_hits,
    const int number_of_tracks_event,
    const int accumulated_tracks_event,
    const VeloState* velo_states_event,
    VeloUTTracking::HitsSoA *hits_layers_events,
    const uint32_t n_hits_layers_events[VeloUTTracking::n_layers],
    int &n_tracks_past_filter
  ) const;

private:

  const float m_minMomentum =       1.5*Gaudi::Units::GeV;
  const float m_minPT =             0.3*Gaudi::Units::GeV;
  const float m_maxPseudoChi2 =     1280.;
  const float m_yTol =              0.5 * Gaudi::Units::mm;
  const float m_yTolSlope =         0.08;
  const float m_hitTol1 =           6.0 * Gaudi::Units::mm;
  const float m_hitTol2 =           0.8 * Gaudi::Units::mm;
  const float m_deltaTx1 =          0.035;
  const float m_deltaTx2 =          0.018;
  const float m_maxXSlope =         0.350;
  const float m_maxYSlope =         0.300;
  const float m_centralHoleSize =   33. * Gaudi::Units::mm;
  const float m_intraLayerDist =    15.0 * Gaudi::Units::mm;
  const float m_overlapTol =        0.7 * Gaudi::Units::mm;
  const float m_passHoleSize =      40. * Gaudi::Units::mm;
  const int   m_minHighThres =      1;
  const bool  m_printVariables =    false;
  const bool  m_passTracks =        false;
  const bool  m_doTiming =          false;

  // typedef MultiIndexedHitContainer<Hit, UT::Info::kNStations, UT::Info::kNLayers>::HitRange HitRange;

  bool filterTrack(
    const VeloState& state ) const;

  bool getHits(
    int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
    int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
    const int posLayers[4][85],
    VeloUTTracking::HitsSoA *hits_layers,
    const uint32_t n_hits_layers[VeloUTTracking::n_layers],
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
    std::vector<VeloUTTracking::TrackUT>& outputTracks,
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
    const uint32_t n_hits_layers[VeloUTTracking::n_layers],
    int posLayers[4][85] ) const
  {
    
    for(int iStation = 0; iStation < 2; ++iStation){
      for(int iLayer = 0; iLayer < 2; ++iLayer){
        int layer = 2*iStation + iLayer;
       	int layer_offset = hits_layers->layer_offset[layer];
	
        size_t pos = 0;
        fillArray( posLayers[layer], 85, pos );

        int bound = -42.0;
        float val = std::copysign(float(bound*bound)/2.0, bound);

        // TODO add bounds checking
        for ( ; pos != n_hits_layers[layer]; ++pos) {
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
          n_hits_layers[layer] );
        
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
    const uint32_t n_hits_layers[VeloUTTracking::n_layers],
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

    size_t pos = posBeg;
    while ( 
      pos <= posEnd && 
      hits_layers->isNotYCompatible( layer_offset + pos, yApprox, m_yTol + m_yTolSlope * std::abs(xTolNormFact) )
    ) { ++pos; }

    const auto xOnTrackProto = myState.x + myState.tx*(zInit - myState.z);
    const auto yyProto =       myState.y - myState.ty*myState.z;

    for (int i=pos; i<posEnd; ++i) {

      const auto xx = hits_layers->xAt( layer_offset + i, yApprox ); 
      const auto dx = xx - xOnTrackProto;
      
      if( dx < -xTolNormFact ) continue;
      if( dx >  xTolNormFact ) break; 

      // -- Now refine the tolerance in Y
      if ( hits_layers->isNotYCompatible( layer_offset + i, yApprox, m_yTol + m_yTolSlope * std::abs(dx*invNormFact)) ) continue;
      
      
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
    const float dz = 0.001*(hit->z - m_zMidUT);
    const float wi = hit->weight2();

    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }

  void addChi2( const float xTTFit, const float xSlopeTTFit, float& chi2 , const VeloUTTracking::Hit* hit) const {
    const float zd    = hit->z;
    const float xd    = xTTFit + xSlopeTTFit*(zd-m_zMidUT);
    const float du    = xd - hit->x;
    chi2 += (du*du)*hit->weight2();
  }

  template <std::size_t N>
  void simpleFit(
    std::array<const VeloUTTracking::Hit*,N>& hits, 
    TrackHelper& helper ) const 
  {
    assert( N==3||N==4 );

    // -- Scale the z-component, to not run into numerical problems
    // -- with floats
    const float zDiff = 0.001*(m_zKink-m_zMidUT);
    float mat[3] = { helper.wb, helper.wb*zDiff, helper.wb*zDiff*zDiff };
    float rhs[2] = { helper.wb* helper.xMidField, helper.wb*helper.xMidField*zDiff };

    const int nHighThres = std::count_if( 
      hits.begin(),  hits.end(), []( const VeloUTTracking::Hit* hit ) { return hit && hit->highThreshold(); });
    
    // -- Veto hit combinations with no high threshold hit
    // -- = likely spillover
    if( nHighThres < m_minHighThres ) return;

    std::for_each( hits.begin(), hits.end(), [&](const VeloUTTracking::Hit* h) { this->addHit(mat,rhs,h); } );

    ROOT::Math::CholeskyDecomp<float, 2> decomp(mat);
    if( !decomp ) return;

    decomp.Solve(rhs);

    const float xSlopeTTFit = 0.001*rhs[1];
    const float xTTFit = rhs[0];

    // new VELO slope x
    const float xb = xTTFit+xSlopeTTFit*(m_zKink-m_zMidUT);
    const float xSlopeVeloFit = (xb-helper.state.x)*helper.invKinkVeloDist;
    const float chi2VeloSlope = (helper.state.tx - xSlopeVeloFit)*m_invSigmaVeloSlope;

    float chi2TT = chi2VeloSlope*chi2VeloSlope;

    std::for_each( hits.begin(), hits.end(), [&](const VeloUTTracking::Hit* h) { this->addChi2(xTTFit,xSlopeTTFit, chi2TT, h); } );

    chi2TT /= (N + 1 - 2);

    if( chi2TT < helper.bestParams[1] ){

      // calculate q/p
      const float sinInX  = xSlopeVeloFit * std::sqrt(1.+xSlopeVeloFit*xSlopeVeloFit);
      const float sinOutX = xSlopeTTFit * std::sqrt(1.+xSlopeTTFit*xSlopeTTFit);
      const float qp = (sinInX-sinOutX);

      helper.bestParams[0] = qp;
      helper.bestParams[1] = chi2TT;
      helper.bestParams[2] = xTTFit;
      helper.bestParams[3] = xSlopeTTFit;

      for ( int i_hit = 0; i_hit < N; ++i_hit ) {
        helper.bestHits[i_hit] = *(hits[i_hit]);
      }
      helper.n_hits = N;
    }

  }

  // ---

  PrUTMagnetTool       m_PrUTMagnetTool;                            ///< Multipupose tool for Bdl and deflection
  float                m_zMidUT;
  float                m_distToMomentum;
  float                m_zKink;
  float                m_sigmaVeloSlope;
  float                m_invSigmaVeloSlope;


};

