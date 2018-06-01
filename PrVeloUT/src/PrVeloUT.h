# pragma once

// #ifndef PRVELOUT_H
// #define PRVELOUT_H 1

//#define SMALL_OUTPUT

#include <cmath>

// Include files
// // from Gaudi
// #include "GaudiAlg/ISequencerTimerTool.h"
// #include "GaudiAlg/Transformer.h"

// // from TrackInterfaces
// #include "TrackInterfaces/ITracksFromTrackR.h"
// #include "Event/Track.h"

// from cuda_hlt/checker/lib
#include "../checker/lib/include/Tracks.h"

// from Rec - PrKernel
#include "include/UTHitHandler.h"
#include "include/UTHitInfo.h"
#include "include/UTHit.h"

// from Gaudi - GaudiKernel
#include "include/DataObject.h"
#include "include/ObjectContainerBase.h"
#include "include/Range.h"

// from LHCb - Tf/TfKernel
#include "include/IndexedHitContainer.h"
#include "include/MultiIndexedHitContainer.h"

// TODO
#include "PrUTMagnetTool.h"

// Math from ROOT
#include "CholeskyDecomp.h"

// Types
#include "include/VeloTypes.h"

// #ifdef SMALL_OUTPUT
// #include "PrKernel/PrVeloUTTrack.h"
// #endif

// #include "vdt/sqrt.h"
/** @class PrVeloUT PrVeloUT.h
   *
   *  PrVeloUT algorithm. This is just a wrapper,
   *  the actual pattern recognition is done in the 'PrVeloUTTool'.
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
   */

struct TrackHelper{
  VeloState state;
  std::array<const Hit*, 4> bestHits = { nullptr, nullptr, nullptr, nullptr};
  std::array<float, 4> bestParams;
  float wb, invKinkVeloDist, xMidField;

  TrackHelper(
    const VeloState& miniState, 
    const float zKink, 
    const float sigmaVeloSlope, 
    const float maxPseudoChi2
  ):
    state(miniState),
    bestParams{{ 0.0, maxPseudoChi2, 0.0, 0.0 }}{
    xMidField = state.x + state.tx*(zKink-state.z);
    const float a = sigmaVeloSlope*(zKink - state.z);
    wb=1./(a*a);
    invKinkVeloDist = 1/(zKink-state.z);
  }
};

class PrVeloUT {

public:
  /// Standard constructor
  // PrVeloUT( const std::string& name, ISvcLocator* pSvcLocator );
  PrVeloUT();
  virtual int initialize() override;    ///< Algorithm initialization
  LHCb::Tracks operator()(const std::vector<Track>& inputTracks) const override;

private:

  float m_minMomentum;    // {this, "minMomentum",      1.5*Gaudi::Units::GeV};
  float m_minPT;          // {this, "minPT",            0.3*Gaudi::Units::GeV};
  float m_maxPseudoChi2;  // {this, "maxPseudoChi2",    1280.};
  float m_yTol;           // {this, "YTolerance",       0.5  * Gaudi::Units::mm}; // 0.8
  float m_yTolSlope;      // {this, "YTolSlope",        0.08}; // 0.2
  float m_hitTol1;        // {this, "HitTol1",          6.0 * Gaudi::Units::mm};
  float m_hitTol2;        // {this, "HitTol2",          0.8 * Gaudi::Units::mm}; // 0.8
  float m_deltaTx1;       // {this, "DeltaTx1",         0.035};
  float m_deltaTx2;       // {this, "DeltaTx2",         0.018}; // 0.02
  float m_maxXSlope;      // {this, "MaxXSlope",        0.350};
  float m_maxYSlope;      // {this, "MaxYSlope",        0.300};
  float m_centralHoleSize;// {this, "centralHoleSize",  33. * Gaudi::Units::mm};
  float m_intraLayerDist; // {this, "IntraLayerDist",   15.0 * Gaudi::Units::mm};
  float m_overlapTol;     // {this, "OverlapTol",       0.7 * Gaudi::Units::mm};
  float m_passHoleSize;   // {this, "PassHoleSize",     40. * Gaudi::Units::mm};
  int   m_minHighThres;   // {this, "MinHighThreshold", 1};
  bool  m_printVariables; // {this, "PrintVariables",   false};
  bool  m_passTracks;     // {this, "PassTracks",       false};
  bool  m_doTiming;       // {this, "TimingMeasurement",false};

  // typedef MultiIndexedHitContainer<Hit, UT::Info::kNStations, UT::Info::kNLayers>::HitRange HitRange;

  bool getState(const Track* iTr, VeloState& trState, Track& outputTracks) const;

  bool getHits(std::array<std::vector<Hit>,4>& hitsInLayers,  const std::array<std::array<HitRange::const_iterator,85>,4>& iteratorsLayers,
               const UT::HitHandler* hh,
               const std::vector<float>& fudgeFactors, VeloState& trState ) const;

  bool formClusters(const std::array<std::vector<Hit>,4>& hitsInLayers, TrackHelper& helper) const;

  void prepareOutputTrack(const Track* veloTrack,
                          const TrackHelper& helper,
                          const std::array<std::vector<Hit>,4>& hitsInLayers,
                          std::vector<Track>& outputTracks,
                          const std::vector<float>& bdlTable) const;

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
  inline void fillIterators(const UT::HitHandler*hh, std::array<std::array<HitRange::const_iterator,85>,4>& iteratorsLayers) const{

    for(int iStation = 0; iStation < 2; ++iStation){
      for(int iLayer = 0; iLayer < 2; ++iLayer){
        const HitRange& hits = hh->hits( iStation, iLayer );

        iteratorsLayers[2*iStation + iLayer].fill( hits.begin() );

        float bound = -42.0;
        float val = std::copysign(bound*bound/2.0, bound);
        const auto itEnd = hits.end();
        for( auto it = hits.begin(); it != itEnd; ++it){

          while( (*it).xAtYEq0() > val){
            iteratorsLayers[2*iStation + iLayer][bound+42] = it;
            ++bound;
            val = std::copysign(bound*bound/2.0, bound);
          }
        }

        std::fill(iteratorsLayers[2*iStation + iLayer].begin() + 42 + int(bound), iteratorsLayers[2*iStation + iLayer].end(), hits.end());

      }
    }
  }

  // ==============================================================================
  // -- Method that finds the hits in a given layer within a certain range
  // ==============================================================================
  inline void findHits( HitRange::const_iterator itH, HitRange::const_iterator itEnd,
                        const VeloState& myState, const float xTolNormFact,
                        const float invNormFact, std::vector<Hits>& hits) const {

    const auto zInit = (*itH).zAtYEq0();
    const auto yApprox = myState.y + myState.ty * (zInit - myState.z);

    while( itH != itEnd && (*itH).isNotYCompatible( yApprox, m_yTol + m_yTolSlope * std::abs(xTolNormFact) )  ) ++itH;

    const auto xOnTrackProto = myState.x + myState.tx*(zInit - myState.z);
    const auto yyProto =       myState.y - myState.ty*myState.z;

    for ( ; itH != itEnd; ++itH ){

      const auto xx = (*itH).xAt(yApprox);
      const auto dx = xx - xOnTrackProto;

      //counter("#findHitsLoop")++;

      if( dx < -xTolNormFact ) continue;
      if( dx >  xTolNormFact ) break;

      // -- Now refine the tolerance in Y
      if(  (*itH).isNotYCompatible( yApprox, m_yTol + m_yTolSlope * std::abs(dx*invNormFact)) ) continue;

      const auto zz = (*itH).zAtYEq0();
      const auto yy = yyProto +  myState.ty*zz;
      const auto xx2 = (*itH).xAt(yy);
      hits.emplace_back(&(*itH), xx2, zz);

    }
  }

  // ===========================================================================================
  // -- 2 helper functions for fit
  // -- Pseudo chi2 fit, templated for 3 or 4 hits
  // ===========================================================================================
  void addHit( float* mat, float* rhs, const Hit* hit)const{
    const float ui = hit->x;
    const float ci = hit->HitPtr->cosT();
    const float dz = 0.001*(hit->z - m_zMidUT);
    const float wi = hit->HitPtr->weight();
    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }

  void addChi2( const float xTTFit, const float xSlopeTTFit, float& chi2 , const Hit* hit)const{
    const float zd    = hit->z;
    const float xd    = xTTFit + xSlopeTTFit*(zd-m_zMidUT);
    const float du    = xd - hit->x;
    chi2 += (du*du)*hit->HitPtr->weight();
  }



  template <std::size_t N>
  void simpleFit( std::array<const Hit*,N> hits, TrackHelper& helper) const {
    assert( N==3||N==4 );

    // -- Scale the z-component, to not run into numerical problems
    // -- with floats
    const float zDiff = 0.001*(m_zKink-m_zMidUT);
    float mat[3] = { helper.wb, helper.wb*zDiff, helper.wb*zDiff*zDiff };
    float rhs[2] = { helper.wb* helper.xMidField, helper.wb*helper.xMidField*zDiff };

    const int nHighThres = std::count_if( hits.begin(),  hits.end(),
                                          []( const Hit* hit ){ return hit && hit->HitPtr->highThreshold(); });


    // -- Veto hit combinations with no high threshold hit
    // -- = likely spillover
    if( nHighThres < m_minHighThres ) return;

    std::for_each( hits.begin(), hits.end(), [&](const auto* h) { this->addHit(mat,rhs,h); } );

    ROOT::Math::CholeskyDecomp<float, 2> decomp(mat);
    if( UNLIKELY(!decomp)) return;

    decomp.Solve(rhs);

    const float xSlopeTTFit = 0.001*rhs[1];
    const float xTTFit = rhs[0];

    // new VELO slope x
    const float xb = xTTFit+xSlopeTTFit*(m_zKink-m_zMidUT);
    const float xSlopeVeloFit = (xb-helper.state.x)*helper.invKinkVeloDist;
    const float chi2VeloSlope = (helper.state.tx - xSlopeVeloFit)*m_invSigmaVeloSlope;

    float chi2TT = chi2VeloSlope*chi2VeloSlope;

    std::for_each( hits.begin(), hits.end(), [&](const auto* h) { this->addChi2(xTTFit,xSlopeTTFit, chi2TT, h); } );

    chi2TT /= (N + 1 - 2);

    if( chi2TT < helper.bestParams[1] ){

      // calculate q/p
      const float sinInX  = xSlopeVeloFit * std::sqrt(1.+xSlopeVeloFit*xSlopeVeloFit);
      const float sinOutX = xSlopeTTFit * std::sqrt(1.+xSlopeTTFit*xSlopeTTFit);
      const float qp = (sinInX-sinOutX);

      helper.bestParams = { qp, chi2TT, xTTFit,xSlopeTTFit };

      std::copy( hits.begin(), hits.end(), helper.bestHits.begin() );
      if( N == 3 ) { helper.bestHits[3] = nullptr ; }
    }

  }
  // --

  // AnyDataHandle<UT::HitHandler> m_HitHandler {UT::Info::HitLocation, Gaudi::DataHandle::Reader, this};

  ITracksFromTrackR*   m_veloUTTool       = nullptr;             ///< The tool that does the actual pattern recognition
  // ISequencerTimerTool* m_timerTool        = nullptr;             ///< Timing tool
  int                  m_veloUTTime       = 0;                   ///< Counter for timing tool
  PrUTMagnetTool*      m_PrUTMagnetTool   = nullptr;             ///< Multipupose tool for Bdl and deflection
  float                m_zMidUT;
  float                m_distToMomentum;
  float                m_zKink;
  float                m_sigmaVeloSlope;
  float                m_invSigmaVeloSlope;


};

// #endif // PRVELOUT_H
