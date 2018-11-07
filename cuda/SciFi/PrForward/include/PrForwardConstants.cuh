#pragma once

/**
   Contains constants needed for the forward tracking
   - cut values
   - geometry descriptions
   - parameterizations

   12/09/2018: cut values are those defined in:
   https://gitlab.cern.ch/lhcb/Rec/blob/master/Tf/TrackSys/python/TrackSys/Configuration.py
   https://gitlab.cern.ch/lhcb/Rec/blob/master/Tf/TrackSys/python/TrackSys/RecoUpgradeTracking.py

   for the RecoFastTrackingStage, using the default values of ConfigHLT1 (master branch of Rec)

 */

#include "VeloEventModel.cuh"

namespace SciFi{
  
  namespace Tracking {

    constexpr int max_candidate_tracks = 5;   // max # of candidate tracks from x hits only
    constexpr int max_tracks_second_loop = 5; // same as above, but for second loop
    constexpr int max_selected_tracks = max_candidate_tracks + max_tracks_second_loop;
    constexpr int max_x_hits = 500;     // max # of hits in all x layers
    constexpr int max_other_hits = 5;   // max # of hits from x planes with more than 1 hit
    constexpr int max_stereo_hits = 25; // max # of hits in all stereo layers
    constexpr int max_coordToFit = 15;  // only for x layers
    constexpr int max_scifi_hits = 20;  // for x and u/v layers
    
    constexpr int nTrackParams = 9;   
    
    constexpr int TMVA_Nvars = 7;
    constexpr int TMVA_Nlayers = 5;

    // Formerly PrParameters
    struct HitSearchCuts {
    __host__ __device__ HitSearchCuts(unsigned int minXHits_, float maxXWindow_,
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
    
    // dump a bunch of options here
    constexpr float        deltaQuality = 0.1; // Difference in quality btw two tracks which share hits when clone killing
    constexpr float        cloneFraction = 0.4; // The fraction of shared SciFi hits btw two tracks to trigger the clone killing
    
    constexpr float        yTolUVSearch           =   11.* Gaudi::Units::mm  ;
    constexpr float        tolY                   =    5.* Gaudi::Units::mm  ;
    constexpr float        tolYSlope              =0.002 * Gaudi::Units::mm  ;
    constexpr float        maxChi2LinearFit       =  100.                    ;   
    constexpr float        maxChi2XProjection     =   15.                    ;   
    constexpr float        maxChi2PerDoF          =    7.                    ;   
    
    constexpr float        tolYMag                =   10.* Gaudi::Units::mm  ;
    constexpr float        tolYMagSlope           =    0.015                 ;   
    constexpr float        minYGap                =  0.4 * Gaudi::Units::mm  ;
    
    constexpr unsigned int minTotalHits           =   10                     ;   
    constexpr float        maxChi2StereoLinear    =   60.                    ;   
    constexpr float        maxChi2Stereo          =    4.5                   ; 
    
    //first loop Hough Cluster search
    constexpr unsigned int minXHits               =    5                     ;   
    constexpr float        maxXWindow             =   1. * Gaudi::Units::mm  ;   //1.2 * Gaudi::Units::mm  ;
    constexpr float        maxXWindowSlope        =0.002 * Gaudi::Units::mm  ;
    constexpr float        maxXGap                =  1. * Gaudi::Units::mm  ;    //1.2 * Gaudi::Units::mm  ;
    constexpr unsigned int minSingleHits          =    2                     ; 
    
    //second loop Hough Cluster search
    constexpr bool         secondLoop             =  true                    ;   
    constexpr unsigned int minXHits_2nd           =    4                     ;   
    constexpr float        maxXWindow_2nd         =  1.5 * Gaudi::Units::mm  ;
    constexpr float        maxXWindowSlope_2nd    =0.002 * Gaudi::Units::mm  ;
    constexpr float        maxXGap_2nd            =  0.5 * Gaudi::Units::mm  ;
    
    //collectX search
    constexpr float        minPt                  =  400 * Gaudi::Units::MeV ;    // 500 * Gaudi::Units::MeV ;
    //stereo hit matching
    constexpr float        tolYCollectX           =   3.5 * Gaudi::Units::mm ;    //4.1* Gaudi::Units::mm ;
    constexpr float        tolYSlopeCollectX      = 0.001 * Gaudi::Units::mm ;    //0.0018 * Gaudi::Units::mm ;
    constexpr float        tolYTriangleSearch     =    20.f                  ;   
    //veloUT momentum estimate
    constexpr bool         useMomentumEstimate    = true                     ;   
    constexpr bool         useWrongSignWindow     = true                     ;   
    constexpr float        wrongSignPT            = 2000.* Gaudi::Units::MeV ; 
    //Track Quality NN
    constexpr float        maxQuality             =   0.9                    ;   
    constexpr float        deltaQuality_NN        =   0.1                    ; 

    // parameterizations
    constexpr float        byParams               = -0.667996;
    constexpr float        cyParams               = -3.68424e-05;

    // z Reference plane
    constexpr float        zReference             = 8520.; // in T2
    
    // TODO: CHECK THESE VALUES USING FRAMEWORK
    constexpr float        xLim_Max               = 3300.;
    constexpr float        yLim_Max               = 2500.;
    constexpr float        xLim_Min               = -3300.;
    constexpr float        yLim_Min               = -25.;

     // TO BE READ FROM XML EVENTUALLY
    constexpr float              magscalefactor         = -1;

    
    struct Arrays {
      // the Magnet Parametrization
      // parameterized in offset [0], (slope difference due to kick)^2 [1],
      // tx^2 [2], ty^2 [3]
      const float        zMagnetParams[4]       = {5212.38, 406.609, -1102.35, -498.039};
      
      // more Parametrizations
      const float        xParams[2]             = {18.6195, -5.55793};
      
      // momentum Parametrization
      const float        momentumParams[6]      = {1.21014, 0.637339, -0.200292, 0.632298, 3.23793, -27.0259};
      
      // covariance values
      const float        covarianceValues[5]    = {4.0, 400.0, 4.e-6, 1.e-4, 0.1};
      
      // definition of zones
      // access upper with offset of 6
      const int	     zoneoffsetpar	      = 6;
      const int        xZones[12]             = {0 , 6 , 8 , 14 , 16 , 22, 1 , 7 , 9 , 15 , 17 , 23 };
      const int        uvZones[12]            = {2 , 4 , 10, 12 , 18 , 20, 3 , 5 , 11, 13 , 19 , 21 };
      
      // ASSORTED GEOMETRY VALUES, eventually read this from some xml
      const float        xZone_zPos[6]          = {7826., 8036., 8508., 8718., 9193., 9403.};
      const float        uvZone_zPos[12]        = {7896., 7966., 8578., 8648., 9263., 9333., 7896., 7966., 8578., 8648., 9263., 9333.};
      const float        uvZone_dxdy[12]        = {0.0874892, -0.0874892, 0.0874892, -0.0874892, 0.0874892, -0.0874892, 0.0874892, -0.0874892, 0.0874892, -0.0874892, 0.0874892, -0.0874892};
      const float        Zone_dzdy[24]          = {0.0036010};
    };

    // Track object used for finding tracks, not the final container for storing the tracks
    struct Track {
      int hit_indices[max_scifi_hits];
      float qop;
      int hitsNum = 0;
      float quality;
      float chi2;
      // [0]: xRef
      // [1]: (xRef-xMag)/(zRef-zMag)
      // [2]: xParams[0] * dSlope
      // [3]: xParams[1] * dSlope
      // [4]:
      // [5]:
      // [6]:
      // [7]: chi2
      // [8]: nDoF
      float trackParams[SciFi::Tracking::nTrackParams];
      Velo::State state_endvelo;
     
      __host__  __device__ void addHit( unsigned int hit ) {
        assert( hitsNum < max_scifi_hits - 1 );
        hit_indices[hitsNum++] = hit;
      }
      
      __host__ __device__ void set_qop( float _qop ) {
        qop = _qop;
      }
    };
    
  } // Tracking
} // SciFi
