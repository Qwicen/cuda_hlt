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
   */

class PrForward {

public:

  // std::vector<std::string> GetFieldMaps(;
  std::vector<VeloUTTracking::TrackVeloUT> operator()(
    const std::vector<VeloUTTracking::TrackVeloUT>& inputTracks,
    ForwardTracking::HitsSoAFwd *hits_layers_events,
    const uint32_t n_hits_layers_events[ForwardTracking::n_layers]
  ) const;

  void setMagScaleFactor(int magscalefactor){m_magscalefactor = magscalefactor;}

private:

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
  const float        m_xLim_Max               = 2700.;
  const float        m_yLim_Max               = 2417.5;
  const float        m_xLim_Min               = 0.;
  const float        m_yLim_Min               = 0.;

  // other variables which get initialized
  float              m_xHits[8]               = {0.,0.,0.,0.,0.,0.,0.,0.};

  float              m_magscalefactor         = 1;

  // What follows is a bunch of cached information from the end velo state which
  // is used to avoid passing it around or regenerating it in all the helper functions
  mutable float           m_x0     = 0.0;
  mutable float           m_y0     = 0.0;
  mutable float           m_z0     = 0.0;
  mutable float           m_tx     = 0.0;
  mutable float           m_ty     = 0.0;
  mutable float           m_qOverP = 0.0;
  mutable float           m_tx2    = 0.0;
  mutable float           m_ty2    = 0.0;
  mutable float           m_slope2 = 0.0;

  // More caching of current values of search windows
  mutable unsigned int    m_minXHits_cur;   // current value for the minimal number of X hits.
  mutable float           m_maxXWindow_cur;
  mutable float           m_maxXWindowSlope_cur;
  mutable float           m_maxXGap_cur;
  mutable int             m_minStereoHits_cur;

  // Vectors of selected hits
  mutable ForwardTracking::HitsSoAFwd  m_hits_layers;
  
  void prepareOutputTrack(  
    const VeloUTTracking::TrackVeloUT& veloUTTrack,
    std::vector<VeloUTTracking::TrackVeloUT>& outputTracks
  ) const;

  // The following functions all implicitly operate on the cached VELO track parameters above
  inline float xFromVelo( const float z )  const { return m_x0 + (z-m_z0) * m_tx; }
  inline float yFromVelo( const float z )  const { return m_y0 + (z-m_z0) * m_ty; }

  std::vector<float> getTrackParameters ( float xAtRef) const {

    float dSlope  = ( xFromVelo(m_zReference) - xAtRef ) / ( m_zReference - m_zMagnetParams[0]);
    const float zMagSlope = m_zMagnetParams[2] * m_tx2 +  m_zMagnetParams[3] * m_ty2;
    const float zMag    = m_zMagnetParams[0] + m_zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
    const float xMag    = xFromVelo( zMag );
    const float slopeT  = ( xAtRef - xMag ) / ( m_zReference - zMag );
    dSlope        = slopeT - m_tx;
    const float dyCoef  = dSlope * dSlope * m_ty;

    std::vector<float> toreturn =  {xAtRef,
                                    slopeT,
                                    1.e-6f * m_xParams[0] * dSlope,
                                    1.e-9f * m_xParams[1] * dSlope,
                                    yFromVelo( m_zReference ),
                                    m_ty + dyCoef * m_byParams,
                                    dyCoef * m_cyParams,
				    0.0,
				    0.0 }; // last elements are chi2 and ndof, as float 
    return toreturn;
  }

  float calcqOverP ( float bx ) const {

    float qop(1.0f/Gaudi::Units::GeV) ;
    float magscalefactor = m_magscalefactor; 
    float bx2  = bx * bx;
    float coef = ( m_momentumParams[0] +
                   m_momentumParams[1] * bx2 +
                   m_momentumParams[2] * bx2 * bx2 +
                   m_momentumParams[3] * bx * m_tx +
                   m_momentumParams[4] * m_ty2 +
                   m_momentumParams[5] * m_ty2 * m_ty2 );
    float proj = sqrt( ( 1.f + m_slope2 ) / ( 1.f + m_tx2 ) ); 
    qop = ( m_tx - bx ) / ( coef * Gaudi::Units::GeV * proj * magscalefactor) ;
    return qop ;

  }

  float zMagnet() {
    
    return ( m_zMagnetParams[0] +
             m_zMagnetParams[2] * m_tx2 +
             m_zMagnetParams[3] * m_ty2 );
  }

  void covariance ( VeloUTTracking::FullState& state, const float qOverP ) {
     
    state.c00 = m_covarianceValues[0];
    state.c11 = m_covarianceValues[1];
    state.c22 = m_covarianceValues[2];
    state.c33 = m_covarianceValues[3];
    state.c44 = m_covarianceValues[4] * qOverP * qOverP;
  }

  float calcDxRef(float pt) const {
    return 3973000. * sqrt( m_slope2 ) / pt - 2200. *  m_ty2 - 1000. * m_tx2; // tune this window
  }

  float straightLineExtend(const float params[4], const float z) const {
    float dz = z - m_zReference;
    return params[0] + dz*(params[1]+dz*(params[2] + dz*params[3]));
  }

  bool isInside(float val, const float min, const float max) {
    return (val > min) && (val < max) ;
  }

  int getLowerBound(float range[],float value,int start, int end){
    int i = start;
    for (i; i<end; i++) {
      if (range[i] > value) break;
    }
    return i;
  }

  bool isValid(int value) {
    return !m_hits_layers.m_used[value];
  }

  int nbDifferent(int planelist[]) {
    int different = 0;
    for (auto i : planelist){different += i > 0 ? 1 : 0;}
    return different;
  }

  int nbSingle(int planelist[]){
    int single = 0;
    for (int i=0;i<12;++i){single += planelist[i] == 1 ? 1 : 0;}
    return single;
  }

  inline float trackToHitDistance( std::vector<float> trackParameters, int hit ) const {
    float z_Hit = m_hits_layers.m_z[hit] + 
		  m_hits_layers.m_dzdy[hit]*straightLineExtend({trackParameters[4],
                                           		        trackParameters[5],
                                        			trackParameters[6],
                                        		 	0.}, m_hits_layers.m_z[hit]);
    float x_track = straightLineExtend({trackParameters[0],
					trackParameters[1],
					trackParameters[2],
					trackParameters[3]}, z_Hit);
    float y_track = straightLineExtend({trackParameters[4],
					trackParameters[5],
					trackParameters[6],
					0.}, z_Hit);
    return m_hits_layers.m_x[hit] + y_track*m_hits_layers.m_dxdy[hit] - x_track; 
  }

  void incrementLineFitParameters(ForwardTracking::LineFitterPars &parameters, int it) {
    float c = m_hits_layers.m_coord[it];
    float w = m_hits_layers.m_w[it];
    float z = m_hits_layers.m_z[it] - parameters.m_z0;
    parameters.m_s0   += w;
    parameters.m_sz   += w * z;
    parameters.m_sz2  += w * z * z;
    parameters.m_sc   += w * c;
    parameters.m_scz  += w * c * z;
  } 

  float getLineFitDistance(ForwardTracking::LineFitterPars &parameters, int it ) { 
    return m_hits_layers.m_coord[it] - (parameters.m_c0 + (m_hits_layers.m_z[it] - parameters.m_z0) * parameters.m_tc);
  }

  float getLineFitChi2(ForwardTracking::LineFitterPars &parameters, int it) {
    float d = getLineFitDistance( hit ); 
    return d * d * m_hits_layers.m_coord[it]; 
  }

  void solveLineFit(ForwardTracking::LineFitterPars &parameters) {
    float den = (parameters.m_sz*parameters.m_sz-parameters.m_s0*parameters.m_sz2);
    parameters.m_c0  = (parameters.m_scz * parameters.m_sz - parameters.m_sc * parameters.m_sz2) / den;
    parameters.m_tc  = (parameters.m_sc *  parameters.m_sz - parameters.m_s0 * parameters.m_scz) / den;
  }

  bool compareByCoord(int i1, int i2) {
    return m_hits_layers.m_coord[i1] < m_hits_layers.m_coord[i2]; 
  }

  void collectAllXHits(std::vector<int>& allXHits, 
		       const float m_xParams_seed[4],
                       const float m_yParams_seed[4],
                       int side) const;

  void xAtRef_SamePlaneHits(std::vector<int>& allXHits,
			    const float m_xParams_seed[4],
 			    int itH, int itEnd) const;
};

