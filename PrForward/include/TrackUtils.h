#pragma once

#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.h"

/**
   Helper functions related to track properties
 */

namespace SciFi {
  namespace Tracking {
    // Formerly PrParameters
    struct HitSearchCuts {
    HitSearchCuts(unsigned int minXHits_, float maxXWindow_,
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
inline float xFromVelo( const float z, FullState state_at_endvelo ) { 
  return state_at_endvelo.x + (z-state_at_endvelo.z) * state_at_endvelo.tx; 
}

// extrapolate y position from given state to z
inline float yFromVelo( const float z, FullState state_at_endvelo ) { 
  return state_at_endvelo.y + (z-state_at_endvelo.z) * state_at_endvelo.ty; 
}

// params[0] = x/y, params[1] = tx/ty
inline float straightLineExtend(const float params[4], float z) {
  float dz = z - SciFi::Tracking::zReference;
  return params[0] + (params[1]+(params[2] + params[3]*dz)*dz)*dz;
}

std::vector<float> getTrackParameters ( float xAtRef, FullState state_at_endvelo);

float calcqOverP ( float bx, FullState state_at_endvelo );

float zMagnet(FullState state_at_endvelo);

void covariance ( FullState& state, const float qOverP );

float calcDxRef(float pt, FullState state_at_endvelo);

float trackToHitDistance( std::vector<float> trackParameters, SciFi::HitsSoA* hits_layers, int hit );

static inline bool lowerByQuality(SciFi::Track t1, SciFi::Track t2) {
  return t1.quality < t2.quality;
}
