#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

#include "Logger.h"

#include "TMVA_MLP_Forward1stLoop.h"
#include "TMVA_MLP_Forward2ndLoop.h"
#include "SystemOfUnits.h"

#include "VeloUTDefinitions.cuh"
#include "ForwardDefinitions.cuh"
#include "PrForwardConstants.h"

/** @class PrForward PrForward.h
   *
   *  - InputTracksName: Input location for VeloUT tracks
   *  - OutputTracksName: Output location for Forward tracks
   *  Based on code written by
   *  2012-03-20 : Olivier Callot
   *  2013-03-15 : Thomas Nikodem
   *  2015-02-13 : Sevda Esen [additional search in the triangles by Marian Stahl]
   *  2016-03-09 : Thomas Nikodem [complete restructuring]
   */

struct PrParameters {
  PrParameters(unsigned int minXHits_, float maxXWindow_,
               float maxXWindowSlope_, float maxXGap_,
               unsigned int minStereoHits_)
    : minXHits{minXHits_}, maxXWindow{maxXWindow_},
    maxXWindowSlope{maxXWindowSlope_}, maxXGap{maxXGap_},
    minStereoHits{minStereoHits_} {}
  const unsigned int minXHits;
  const float        maxXWindow;
  const float        maxXWindowSlope;
  const float        maxXGap;
  unsigned int       minStereoHits;
};

class PrForward {

public:

  PrForward();

  std::vector<ForwardTracking::TrackForward> operator()(
    const std::vector<VeloUTTracking::TrackVeloUT>& inputTracks,
    ForwardTracking::HitsSoAFwd *hits_layers_events
  ) const;

  
private:

  const std::vector<std::string> mlpInputVars {{"nPlanes"}, {"dSlope"}, {"dp"}, {"slope2"}, {"dby"}, {"dbx"}, {"day"}};

 
  // Vectors of selected hits
  mutable ForwardTracking::HitsSoAFwd  m_hits_layers;
 
  ReadMLP_Forward1stLoop m_MLPReader_1st;
  ReadMLP_Forward2ndLoop m_MLPReader_2nd;
 
  void find_forward_tracks(  
    const VeloUTTracking::TrackVeloUT& veloUTTrack,
    std::vector<ForwardTracking::TrackForward>& outputTracks
  ) const;

  // The following functions all implicitly operate on the cached VELO track parameters above
  inline float xFromVelo( const float z, FullState state_at_endvelo )  const { 
    return state_at_endvelo.x + (z-state_at_endvelo.z) * state_at_endvelo.tx; 
  }
  inline float yFromVelo( const float z, FullState state_at_endvelo )  const { 
    return state_at_endvelo.y + (z-state_at_endvelo.z) * state_at_endvelo.ty; 
  }

  std::vector<float> getTrackParameters ( float xAtRef, FullState state_at_endvelo) const {

    float dSlope  = ( xFromVelo(Forward::zReference,state_at_endvelo) - xAtRef ) / ( Forward::zReference - Forward::zMagnetParams[0]);
    const float zMagSlope = Forward::zMagnetParams[2] * pow(state_at_endvelo.tx,2) +  Forward::zMagnetParams[3] * pow(state_at_endvelo.ty,2);
    const float zMag    = Forward::zMagnetParams[0] + Forward::zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
    const float xMag    = xFromVelo( zMag, state_at_endvelo );
    const float slopeT  = ( xAtRef - xMag ) / ( Forward::zReference - zMag );
    dSlope        = slopeT - state_at_endvelo.tx;
    const float dyCoef  = dSlope * dSlope * state_at_endvelo.ty;

    std::vector<float> toreturn =  {xAtRef,
                                    slopeT,
                                    1.e-6f * Forward::xParams[0] * dSlope,
                                    1.e-9f * Forward::xParams[1] * dSlope,
                                    yFromVelo( Forward::zReference, state_at_endvelo ),
                                    state_at_endvelo.ty + dyCoef * Forward::byParams,
                                    dyCoef * Forward::cyParams,
				    0.0,
				    0.0 }; // last elements are chi2 and ndof, as float 
    return toreturn;
  }

  float calcqOverP ( float bx, FullState state_at_endvelo ) const {

    float qop(1.0f/Gaudi::Units::GeV) ;
    float bx2  = bx * bx;
    float coef = ( Forward::momentumParams[0] +
                   Forward::momentumParams[1] * bx2 +
                   Forward::momentumParams[2] * bx2 * bx2 +
                   Forward::momentumParams[3] * bx * state_at_endvelo.tx +
                   Forward::momentumParams[4] * pow(state_at_endvelo.ty,2) +
                   Forward::momentumParams[5] * pow(state_at_endvelo.ty,2) * pow(state_at_endvelo.ty,2) );
    float m_slope2 = pow(state_at_endvelo.tx,2) + pow(state_at_endvelo.ty,2);
    float proj = sqrt( ( 1.f + m_slope2 ) / ( 1.f + pow(state_at_endvelo.tx,2) ) ); 
    qop = ( state_at_endvelo.tx - bx ) / ( coef * Gaudi::Units::GeV * proj * Forward::magscalefactor) ;
    return qop ;

  }

  // DvB: what does this do?
  // -> get position within magnet (?)
  float zMagnet(FullState state_at_endvelo) const {
    
    return ( Forward::zMagnetParams[0] +
             Forward::zMagnetParams[2] * pow(state_at_endvelo.tx,2) +
             Forward::zMagnetParams[3] * pow(state_at_endvelo.ty,2) );
  }

  void covariance ( FullState& state, const float qOverP ) const {
     
    state.c00 = Forward::covarianceValues[0];
    state.c11 = Forward::covarianceValues[1];
    state.c22 = Forward::covarianceValues[2];
    state.c33 = Forward::covarianceValues[3];
    state.c44 = Forward::covarianceValues[4] * qOverP * qOverP;
  }

  float calcDxRef(float pt, FullState state_at_endvelo) const {
    float m_slope2 = pow(state_at_endvelo.tx,2) + pow(state_at_endvelo.ty,2);
    return 3973000. * sqrt( m_slope2 ) / pt - 2200. *  pow(state_at_endvelo.ty,2) - 1000. * pow(state_at_endvelo.tx,2); // tune this window
  }

  // params[0] = x/y, params[1] = tx/ty
  float straightLineExtend(const float params[4], float z) const {
    float dz = z - Forward::zReference;
    return params[0] + (params[1]+(params[2] + params[3]*dz)*dz)*dz;
  }

  bool isInside(float val, const float min, const float max) const {
    return (val > min) && (val < max) ;
  }

  int getLowerBound(float range[],float value,int start, int end) const {
    int i = start;
    for (; i<end; i++) {
      if (range[i] > value) break;
    }
    return i;
  }

  bool isValid(int value) const {
    return !m_hits_layers.m_used[value];
  }

  int nbDifferent(int planelist[]) const {
    int different = 0;
    for (int i=0;i<12;++i){different += planelist[i] > 0 ? 1 : 0;}
    return different;
  }

  int nbSingle(int planelist[]) const {
    int single = 0;
    for (int i=0;i<12;++i){single += planelist[i] == 1 ? 1 : 0;}
    return single;
  }

  inline float trackToHitDistance( std::vector<float> trackParameters, int hit ) const {
    const float parsX[4] = {trackParameters[0],
                           trackParameters[1],
                           trackParameters[2],
                           trackParameters[3]};
    const float parsY[4] = {trackParameters[4],
                           trackParameters[5],
                           trackParameters[6],
                           0.}; 
    float z_Hit = m_hits_layers.m_z[hit] + 
		  m_hits_layers.m_dzdy[hit]*straightLineExtend(parsY, m_hits_layers.m_z[hit]);
    float x_track = straightLineExtend(parsX,z_Hit);
    float y_track = straightLineExtend(parsY,z_Hit);
    return m_hits_layers.m_x[hit] + y_track*m_hits_layers.m_dxdy[hit] - x_track; 
  }

  void incrementLineFitParameters(ForwardTracking::LineFitterPars &parameters, int it) const {
    float c = m_hits_layers.m_coord[it];
    float w = m_hits_layers.m_w[it];
    float z = m_hits_layers.m_z[it] - parameters.m_z0;
    parameters.m_s0   += w;
    parameters.m_sz   += w * z;
    parameters.m_sz2  += w * z * z;
    parameters.m_sc   += w * c;
    parameters.m_scz  += w * c * z;
  } 

  float getLineFitDistance(ForwardTracking::LineFitterPars &parameters, int it ) const { 
    return m_hits_layers.m_coord[it] - (parameters.m_c0 + (m_hits_layers.m_z[it] - parameters.m_z0) * parameters.m_tc);
  }

  float getLineFitChi2(ForwardTracking::LineFitterPars &parameters, int it) const {
    float d = getLineFitDistance( parameters, it ); 
    return d * d * m_hits_layers.m_coord[it]; 
  }

  void solveLineFit(ForwardTracking::LineFitterPars &parameters) const {
    float den = (parameters.m_sz*parameters.m_sz-parameters.m_s0*parameters.m_sz2);
    parameters.m_c0  = (parameters.m_scz * parameters.m_sz - parameters.m_sc * parameters.m_sz2) / den;
    parameters.m_tc  = (parameters.m_sc *  parameters.m_sz - parameters.m_s0 * parameters.m_scz) / den;
  }

  static bool lowerByQuality(ForwardTracking::TrackForward t1, ForwardTracking::TrackForward t2) {
    return t1.quality < t2.quality;
  }

  void collectAllXHits(std::vector<int>& allXHits, 
		       const float xParams_seed[4],
                       const float yParams_seed[4],
		       FullState state_at_endvelo,
                       int side) const;

  void selectXCandidates(std::vector<int>& allXHits,
                         const VeloUTTracking::TrackVeloUT& veloUTTrack,
                         std::vector<ForwardTracking::TrackForward>& outputTracks,
                         const float zRef_track,
                         const float xParams_seed[4],
                         const float yParams_seed[4],
			 FullState state_at_endvelo,
		         PrParameters& pars_cur,
                         int side) const;

  bool addHitsOnEmptyXLayers(std::vector<float> &trackParameters,
                             const float xParams_seed[4],
                             const float yParams_seed[4],
                             bool fullFit,
                             std::vector<unsigned int> &pc,
                             int planelist[],
			     PrParameters& pars_cur,
                             int side) const;

  bool fitXProjection(std::vector<float> &trackParameters,
                      std::vector<unsigned int> &pc,
                      int planelist[],
		      PrParameters& pars_cur) const;

  void fastLinearFit(std::vector<float> &trackParameters,
                     std::vector<unsigned int> &pc,
                     int planelist[],
	             PrParameters& pars_cur) const;

  void selectFullCandidates(std::vector<ForwardTracking::TrackForward>& outputTracks,
                            const float xParams_seed[4],
                            const float yParams_seed[4],
			    FullState state_at_endvelo,
			    PrParameters& pars_cur) const;

  bool selectStereoHits(ForwardTracking::TrackForward& track,
                        std::vector<int> stereoHits,
		        FullState state_at_endvelo,
		        PrParameters& pars_cur) const;

  bool addHitsOnEmptyStereoLayers(ForwardTracking::TrackForward& track,
                                  std::vector<int>& stereoHits,
                                  std::vector<unsigned int> &pc,
                                  int planelist[],
  				  FullState state_at_endvelo,
			          PrParameters& pars_cur) const;

  bool fitYProjection(ForwardTracking::TrackForward& track,
                      std::vector<int>& stereoHits,
                      std::vector<unsigned int> &pc,
                      int planelist[],
		      FullState state_at_endvelo,
		      PrParameters& pars_cur) const;

  std::vector<int> collectStereoHits(ForwardTracking::TrackForward& track,
				     FullState state_at_endvelo,
				     PrParameters& pars_cur) const;

  void xAtRef_SamePlaneHits(std::vector<int>& allXHits,
			    const float xParams_seed[4],
			    FullState state_at_endvelo,
 			    int itH, int itEnd) const;
};

