#pragma once

#include <cmath>

#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.h"
#include "PrVeloUT.cuh" 

/**
   Helper functions related to track properties
 */

namespace SciFi {
  namespace Tracking {
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

    struct LineFitterPars {
      float   m_z0 = 0.; 
      float   m_c0 = 0.; 
      float   m_tc = 0.; 
      
      float m_s0 = 0.; 
      float m_sz = 0.; 
      float m_sz2 = 0.; 
      float m_sc = 0.; 
      float m_scz = 0.;   
    };
  } // Tracking
} // SciFi

// extrapolate x position from given state to z
__host__ __device__ inline float xFromVelo( const float z, MiniState velo_state ) { 
  return velo_state.x + (z-velo_state.z) * velo_state.tx; 
}

// extrapolate y position from given state to z
__host__ __device__ inline float yFromVelo( const float z, MiniState velo_state ) { 
  return velo_state.y + (z-velo_state.z) * velo_state.ty; 
}

// params[0] = x/y, params[1] = tx/ty
__host__ __device__ inline float straightLineExtend(const float params[4], float z) {
  float dz = z - SciFi::Tracking::zReference;
  return params[0] + (params[1]+(params[2] + params[3]*dz)*dz)*dz;
}

__host__ __device__ void getTrackParameters ( float xAtRef, MiniState velo_state, float trackParams[SciFi::Tracking::nTrackParams]);

__host__ __device__ float calcqOverP ( float bx, MiniState velo_state );

__host__ __device__ float zMagnet(MiniState velo_state);

__host__ __device__ void covariance ( FullState& state, const float qOverP );

__host__ __device__ float calcDxRef(float pt, MiniState velo_state);

__host__ __device__ float trackToHitDistance( float trackParameters[SciFi::Tracking::nTrackParams], SciFi::HitsSoA* hits_layers, int hit );

__host__ __device__ static inline bool lowerByQuality(SciFi::Tracking::Track t1, SciFi::Tracking::Track t2) {
  return t1.quality < t2.quality;
}

__host__ __device__ float chi2XHit( const float parsX[4], SciFi::HitsSoA* hits_layers, const int hit );
