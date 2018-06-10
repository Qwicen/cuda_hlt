#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

//#include "../../PrUTMagnetTool/PrUTMagnetTool.h"
#include "../../main/include/Logger.h"

// Math from ROOT
#include "../include/CholeskyDecomp.h"

#include "../include/VeloTypes.h"
#include "../include/SystemOfUnits.h"

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
   *  2018-05-05: Plácido Fernández (make standalone)
   */

// TODO Fake MagnetTool
struct PrUTMagnetTool {
  //const float m_zMidUT = 0.0;
  //const float m_averageDist2mom = 0.0;
  std::vector<float> dxLayTable;
  std::vector<float> bdlTable;

  PrUTMagnetTool(){}
  PrUTMagnetTool( const std::vector<float> _dxLayTable, const std::vector<float> _bdlTable ) : dxLayTable(_dxLayTable), bdlTable(_bdlTable) {}
  
  //float zMidUT() { return m_zMidUT; }
  //float averageDist2mom() { return m_averageDist2mom; }
  std::vector<float> returnDxLayTable() const { return dxLayTable; }
  std::vector<float> returnBdlTable() const { return bdlTable; }
};

struct TrackHelper{
  VeloState state;
  std::array<const VeloUTTracking::Hit*, 4> bestHits = { nullptr, nullptr, nullptr, nullptr};
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

  std::vector<std::string> GetFieldMaps();
  virtual int initialize();
  std::vector<VeloUTTracking::TrackUT> operator()(const std::vector<VeloUTTracking::TrackVelo>& inputTracks, const std::array<std::vector<VeloUTTracking::Hit>,4> &inputHits) const;

private:

  const float m_minMomentum = 1.5*Gaudi::Units::GeV;
  const float m_minPT = 0.3*Gaudi::Units::GeV;
  const float m_maxPseudoChi2 = 1280.;
  const float m_yTol = 0.5  * Gaudi::Units::mm;
  const float m_yTolSlope = 0.08;
  const float m_hitTol1 = 6.0 * Gaudi::Units::mm;
  const float m_hitTol2 = 0.8 * Gaudi::Units::mm;
  const float m_deltaTx1 = 0.035;
  const float m_deltaTx2 = 0.018;
  const float m_maxXSlope = 0.350;
  const float m_maxYSlope = 0.300;
  const float m_centralHoleSize = 33. * Gaudi::Units::mm;
  const float m_intraLayerDist = 15.0 * Gaudi::Units::mm;
  const float m_overlapTol = 0.7 * Gaudi::Units::mm;
  const float m_passHoleSize = 40. * Gaudi::Units::mm;
  const int   m_minHighThres = 1;
  const bool  m_printVariables = false;
  const bool  m_passTracks = false;
  const bool  m_doTiming = false;

  // typedef MultiIndexedHitContainer<Hit, UT::Info::kNStations, UT::Info::kNLayers>::HitRange HitRange;

  bool getState(
    const VeloUTTracking::TrackVelo& iTr, 
    VeloState& trState ) const;

  bool getHits(
    std::array<std::vector<VeloUTTracking::Hit>,4>& hitsInLayers, 
    const std::array<std::vector<VeloUTTracking::Hit>,4>& inputHits,
    const std::vector<float>& fudgeFactors, 
    const VeloState& trState ) const;

  bool formClusters(const std::array<std::vector<VeloUTTracking::Hit>,4>& hitsInLayers, TrackHelper& helper) const;

  void prepareOutputTrack(const VeloUTTracking::TrackVelo& veloTrack,
                          const TrackHelper& helper,
                          const std::array<std::vector<VeloUTTracking::Hit>,4>& hitsInLayers,
                          std::vector<VeloUTTracking::TrackUT>& outputTracks,
                          const std::vector<float>& bdlTable) const;

  // ==============================================================================
  // -- Method that finds the hits in a given layer within a certain range
  // ==============================================================================
  inline void findHits( 
    const std::vector<VeloUTTracking::Hit>& inputHits,
    const VeloState& myState, 
    const float xTolNormFact,
    const float invNormFact,
    std::vector<VeloUTTracking::Hit>& outHits ) const 
  {
    const auto zInit = inputHits.at(0).zAtYEq0();
    const auto yApprox = myState.y + myState.ty * (zInit - myState.z);

    int pos = 0;
    for (auto& hit : inputHits) {
      if ( hit.isNotYCompatible(yApprox, m_yTol + m_yTolSlope * std::abs(xTolNormFact)) ) {
        ++pos;
      }
    }
    //debug_cout << "position =  " << pos << ", size = " << inputHits.size() << std::endl;
    
    const auto xOnTrackProto = myState.x + myState.tx*(zInit - myState.z);
    const auto yyProto =       myState.y - myState.ty*myState.z;

    for (int i=pos; i<inputHits.size(); ++i) {

      const VeloUTTracking::Hit& hit = inputHits[pos];

      const auto xx = hit.xAt(yApprox);
      const auto dx = xx - xOnTrackProto;

      // debug_cout << "dx = " << dx << ", xTolNormFact = " << xTolNormFact << std::endl;

      if( dx < -xTolNormFact ) continue;
      if( dx >  xTolNormFact ) continue; // DvB: changed from break;
	    

      // -- Now refine the tolerance in Y
      //debug_cout << "yApprox = " << yApprox << " tol = " << m_yTolSlope * std::abs(dx*invNormFact) << ", invNormFact = " << invNormFact << ", yMin - tol = " << hit.yMin() - m_yTol + m_yTolSlope * std::abs(dx*invNormFact) << std::endl;
      if( hit.isNotYCompatible( yApprox, m_yTol + m_yTolSlope * std::abs(dx*invNormFact)) ) continue;
      //std::cout << "past y criteria " << std::endl;

      const auto zz = hit.zAtYEq0();
      const auto yy = yyProto +  myState.ty*zz;
      const auto xx2 = hit.xAt(yy);

      // TODO avoid the copy - remove the const?
      VeloUTTracking::Hit temp_hit = hit;
      temp_hit.m_second_x = xx2;
      temp_hit.m_second_z = zz;


      outHits.emplace_back(temp_hit);
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
    const float wi = hit->weight();
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
    chi2 += (du*du)*hit->weight();
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

    // TODO uncomment
    // const int nHighThres = std::count_if( hits.begin(),  hits.end(),
    //                                       []( VeloUTTracking::Hit* hit ) { return hit && hit->highThreshold(); });

    const int nHighThres = 2;

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

      helper.bestParams = { qp, chi2TT, xTTFit,xSlopeTTFit };

      std::copy( hits.begin(), hits.end(), helper.bestHits.begin() );
      if( N == 3 ) { helper.bestHits[3] = nullptr ; }
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
