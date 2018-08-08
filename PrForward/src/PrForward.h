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
#include "../../PrVeloUT/include/CholeskyDecomp.h"

#include "../include/TMVA_MLP_Forward1stLoop.h"
#include "../include/TMVA_MLP_Forward2ndLoop.h"
#include "../../PrVeloUT/include/VeloTypes.h"
#include "../../PrVeloUT/include/SystemOfUnits.h"

#include "../../cuda/veloUT/common/include/VeloUTDefinitions.cuh"
#include "../../cuda/forward/common/include/ForwardDefinitions.cuh"

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

  bool initialize(){return true;}

  // std::vector<std::string> GetFieldMaps(;
  std::vector<VeloUTTracking::TrackVeloUT> operator()(
    const std::vector<VeloUTTracking::TrackVeloUT>& inputTracks,
    ForwardTracking::HitsSoAFwd *hits_layers_events
  ) const;

  void setMagScaleFactor(int magscalefactor){m_magscalefactor = magscalefactor;}

private:

  const std::vector<std::string> mlpInputVars {{"nPlanes"}, {"dSlope"}, {"dp"}, {"slope2"}, {"dby"}, {"dbx"}, {"day"}};

  // dump a bunch of options here
  const float        m_deltaQuality = 0.1; // Difference in quality btw two tracks which share hits when clone killing
  const float        m_cloneFraction = 0.4; // The fraction of shared SciFi hits btw two tracks to trigger the clone killing
 
  const float        m_yTolUVSearch           =   11.* Gaudi::Units::mm  ;
  const float        m_tolY                   =    5.* Gaudi::Units::mm  ;
  const float        m_tolYSlope              =0.002 * Gaudi::Units::mm  ;
  const float        m_maxChi2LinearFit       =  100.                    ;   
  const float        m_maxChi2XProjection     =   15.                    ;   
  const float        m_maxChi2PerDoF          =    7.                    ;   

  const float        m_tolYMag                =   10.* Gaudi::Units::mm  ;
  const float        m_tolYMagSlope           =    0.015                 ;   
  const float        m_minYGap                =  0.4 * Gaudi::Units::mm  ;

  const unsigned int m_minTotalHits           =   10                     ;   
  const float        m_maxChi2StereoLinear    =   60.                    ;   
  const float        m_maxChi2Stereo          =    4.5                   ; 
 
  //first loop Hough Cluster search
  const unsigned int m_minXHits               =    5                     ;   
  const float        m_maxXWindow             =  1.2 * Gaudi::Units::mm  ;
  const float        m_maxXWindowSlope        =0.002 * Gaudi::Units::mm  ;
  const float        m_maxXGap                =  1.2 * Gaudi::Units::mm  ;
  const unsigned int m_minSingleHits          =    2                     ; 
 
  //second loop Hough Cluster search
  const bool         m_secondLoop             =  true                    ;   
  const unsigned int m_minXHits_2nd           =    4                     ;   
  const float        m_maxXWindow_2nd         =  1.5 * Gaudi::Units::mm  ;
  const float        m_maxXWindowSlope_2nd    =0.002 * Gaudi::Units::mm  ;
  const float        m_maxXGap_2nd            =  0.5 * Gaudi::Units::mm  ;

  //collectX search
  const float        m_minPt                  =  500 * Gaudi::Units::MeV ;
  //stereo hit matching
  const float        m_tolYCollectX           =    4.1* Gaudi::Units::mm ;
  const float        m_tolYSlopeCollectX      =0.0018 * Gaudi::Units::mm ;
  const float        m_tolYTriangleSearch     =    20.f                  ;   
  //veloUT momentum estimate
  const bool         m_useMomentumEstimate    = true                     ;   
  const bool         m_useWrongSignWindow     = true                     ;   
  const float        m_wrongSignPT            = 2000.* Gaudi::Units::MeV ; 
  //Track Quality NN
  const float        m_maxQuality             =   0.9                    ;   
  const float        m_deltaQuality_NN        =   0.1                    ; 

  // the Magnet Parametrization
  const float        m_zMagnetParams[4]       = {5212.38, 406.609, -1102.35, -498.039};

  // more Parametrizations
  const float        m_xParams[2]             = {18.6195, -5.55793};
  const float        m_byParams               = -0.667996;
  const float        m_cyParams               = -3.68424e-05;
 
  // momentum Parametrization
  const float        m_momentumParams[6]      = {1.21014, 0.637339, -0.200292, 0.632298, 3.23793, -27.0259};
  
  // covariance values
  const float        m_covarianceValues[5]    = {4.0, 400.0, 4.e-6, 1.e-4, 0.1};

  // z Reference plane
  const float        m_zReference             = 8520.;

  // definition of zones
  // access upper with offset of 6
  const int	     m_zoneoffsetpar	      = 6;
  const int          m_xZones[12]             = {0 , 6 , 8 , 14 , 16 , 22, 1 , 7 , 9 , 15 , 17 , 23 };
  const int          m_uvZones[12]            = {2 , 4 , 10, 12 , 18 , 20, 3 , 5 , 11, 13 , 19 , 21 };

  // ASSORTED GEOMETRY VALUES, eventually read this from some xml
  const float        m_xZone_zPos[6]          = {7826., 8036., 8508., 8718., 9193., 9403.};
  const float        m_uvZone_zPos[12]        = {7896., 7966., 8578., 8648., 9263., 9333., 7896., 7966., 8578., 8648., 9263., 9333.};
  const float        m_uvZone_dxdy[12]        = {0.0874892, -0.0874892, 0.0874892, -0.0874892, 0.0874892, -0.0874892, 0.0874892, -0.0874892, 0.0874892, -0.0874892, 0.0874892, -0.0874892};
  const float        m_Zone_dzdy[24]          = {0.0036010};
  // CHECK THESE VALUES USING FRAMEWORK
  const float        m_xLim_Max               = 3300.;
  const float        m_yLim_Max               = 2500.;
  const float        m_xLim_Min               = -3300.;
  const float        m_yLim_Min               = -25.;

  // other variables which get initialized
  float              m_xHits[8]               = {0.,0.,0.,0.,0.,0.,0.,0.};

  // TO BE READ FROM XML EVENTUALLY
  float              m_magscalefactor         = -1;

  // Vectors of selected hits
  mutable ForwardTracking::HitsSoAFwd  m_hits_layers;
 
  ReadMLP_Forward1stLoop m_MLPReader_1st;
  ReadMLP_Forward2ndLoop m_MLPReader_2nd;
 
  void prepareOutputTrack(  
    const VeloUTTracking::TrackVeloUT& veloUTTrack,
    std::vector<VeloUTTracking::TrackVeloUT>& outputTracks
  ) const;

  // The following functions all implicitly operate on the cached VELO track parameters above
  inline float xFromVelo( const float z, VeloUTTracking::FullState state_at_endvelo )  const { 
    return state_at_endvelo.x + (z-state_at_endvelo.z) * state_at_endvelo.tx; 
  }
  inline float yFromVelo( const float z, VeloUTTracking::FullState state_at_endvelo )  const { 
    return state_at_endvelo.y + (z-state_at_endvelo.z) * state_at_endvelo.ty; 
  }

  std::vector<float> getTrackParameters ( float xAtRef, VeloUTTracking::FullState state_at_endvelo) const {

    float dSlope  = ( xFromVelo(m_zReference,state_at_endvelo) - xAtRef ) / ( m_zReference - m_zMagnetParams[0]);
    const float zMagSlope = m_zMagnetParams[2] * pow(state_at_endvelo.tx,2) +  m_zMagnetParams[3] * pow(state_at_endvelo.ty,2);
    const float zMag    = m_zMagnetParams[0] + m_zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
    const float xMag    = xFromVelo( zMag, state_at_endvelo );
    const float slopeT  = ( xAtRef - xMag ) / ( m_zReference - zMag );
    dSlope        = slopeT - state_at_endvelo.tx;
    const float dyCoef  = dSlope * dSlope * state_at_endvelo.ty;

    std::vector<float> toreturn =  {xAtRef,
                                    slopeT,
                                    1.e-6f * m_xParams[0] * dSlope,
                                    1.e-9f * m_xParams[1] * dSlope,
                                    yFromVelo( m_zReference, state_at_endvelo ),
                                    state_at_endvelo.ty + dyCoef * m_byParams,
                                    dyCoef * m_cyParams,
				    0.0,
				    0.0 }; // last elements are chi2 and ndof, as float 
    return toreturn;
  }

  float calcqOverP ( float bx, VeloUTTracking::FullState state_at_endvelo ) const {

    float qop(1.0f/Gaudi::Units::GeV) ;
    float magscalefactor = m_magscalefactor; 
    float bx2  = bx * bx;
    float coef = ( m_momentumParams[0] +
                   m_momentumParams[1] * bx2 +
                   m_momentumParams[2] * bx2 * bx2 +
                   m_momentumParams[3] * bx * state_at_endvelo.tx +
                   m_momentumParams[4] * pow(state_at_endvelo.ty,2) +
                   m_momentumParams[5] * pow(state_at_endvelo.ty,2) * pow(state_at_endvelo.ty,2) );
    float m_slope2 = pow(state_at_endvelo.tx,2) + pow(state_at_endvelo.ty,2);
    float proj = sqrt( ( 1.f + m_slope2 ) / ( 1.f + pow(state_at_endvelo.tx,2) ) ); 
    qop = ( state_at_endvelo.tx - bx ) / ( coef * Gaudi::Units::GeV * proj * magscalefactor) ;
    return qop ;

  }

  float zMagnet(VeloUTTracking::FullState state_at_endvelo) const {
    
    return ( m_zMagnetParams[0] +
             m_zMagnetParams[2] * pow(state_at_endvelo.tx,2) +
             m_zMagnetParams[3] * pow(state_at_endvelo.ty,2) );
  }

  void covariance ( VeloUTTracking::FullState& state, const float qOverP ) const {
     
    state.c00 = m_covarianceValues[0];
    state.c11 = m_covarianceValues[1];
    state.c22 = m_covarianceValues[2];
    state.c33 = m_covarianceValues[3];
    state.c44 = m_covarianceValues[4] * qOverP * qOverP;
  }

  float calcDxRef(float pt, VeloUTTracking::FullState state_at_endvelo) const {
    float m_slope2 = pow(state_at_endvelo.tx,2) + pow(state_at_endvelo.ty,2);
    return 3973000. * sqrt( m_slope2 ) / pt - 2200. *  pow(state_at_endvelo.ty,2) - 1000. * pow(state_at_endvelo.tx,2); // tune this window
  }

  float straightLineExtend(const float params[4], float z) const {
    float dz = z - m_zReference;
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

  static bool lowerByQuality(VeloUTTracking::TrackVeloUT t1, VeloUTTracking::TrackVeloUT t2) {
    return t1.trackForward.quality < t2.trackForward.quality;
  }

  void collectAllXHits(std::vector<int>& allXHits, 
		       const float m_xParams_seed[4],
                       const float m_yParams_seed[4],
		       VeloUTTracking::FullState state_at_endvelo,
                       int side) const;

  void selectXCandidates(std::vector<int>& allXHits,
                         const VeloUTTracking::TrackVeloUT& veloUTTrack,
                         std::vector<VeloUTTracking::TrackVeloUT>& outputTracks,
                         const float m_zRef_track,
                         const float m_xParams_seed[4],
                         const float m_yParams_seed[4],
			 VeloUTTracking::FullState state_at_endvelo,
		         PrParameters& pars_cur,
                         int side) const;

  bool addHitsOnEmptyXLayers(std::vector<float> &trackParameters,
                             const float m_xParams_seed[4],
                             const float m_yParams_seed[4],
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

  void selectFullCandidates(std::vector<VeloUTTracking::TrackVeloUT>& outputTracks,
                            const float m_xParams_seed[4],
                            const float m_yParams_seed[4],
			    VeloUTTracking::FullState state_at_endvelo,
			    PrParameters& pars_cur) const;

  bool selectStereoHits(VeloUTTracking::TrackVeloUT& track,
                        std::vector<int> stereoHits,
		        VeloUTTracking::FullState state_at_endvelo,
		        PrParameters& pars_cur) const;

  bool addHitsOnEmptyStereoLayers(VeloUTTracking::TrackVeloUT& track,
                                  std::vector<int>& stereoHits,
                                  std::vector<unsigned int> &pc,
                                  int planelist[],
  				  VeloUTTracking::FullState state_at_endvelo,
			          PrParameters& pars_cur) const;

  bool fitYProjection(VeloUTTracking::TrackVeloUT& track,
                      std::vector<int>& stereoHits,
                      std::vector<unsigned int> &pc,
                      int planelist[],
		      VeloUTTracking::FullState state_at_endvelo,
		      PrParameters& pars_cur) const;

  std::vector<int> collectStereoHits(VeloUTTracking::TrackVeloUT& track,
				     VeloUTTracking::FullState state_at_endvelo,
				     PrParameters& pars_cur) const;

  void xAtRef_SamePlaneHits(std::vector<int>& allXHits,
			    const float m_xParams_seed[4],
			    VeloUTTracking::FullState state_at_endvelo,
 			    int itH, int itEnd) const;
};

