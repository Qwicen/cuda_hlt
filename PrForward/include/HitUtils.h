#pragma once

#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.h"

/**
   Helper functions related to properties of hits on planes
 */


struct PlaneCounter{
  int planeList[SciFi::Constants::n_layers] = {0};
  unsigned int nbDifferent = 0;

  inline void addHit( int plane ) {
    nbDifferent += (int)( (planeList[plane] += 1 ) == 1) ;
  }

  inline void removeHit( int plane ) {
    nbDifferent -= ((int)( (planeList[plane] -= 1 ) == 0)) ;
  }

  inline int nbInPlane( int plane ) const {
    return planeList[plane];
  }

  inline int nbSingle() const {
    int single = 0;
    for (int i=0; i < SciFi::Constants::n_layers; ++i) {
      single += planeList[i] == 1 ? 1 : 0;
    }
    return single;
  }
 
  inline void clear() {
    nbDifferent = 0;
    for ( int i = 0; i < SciFi::Constants::n_layers; ++i ) {
      planeList[i] = 0;
    }
  }
  
};


// check that val is within [min, max]
inline bool isInside(float val, const float min, const float max) {
  return (val > min) && (val < max) ;
}

// get lowest index where range[index] > value, within [start,end] of range 
inline int getLowerBound(float range[],float value,int start, int end) {
  int i = start;
  for (; i<end; i++) {
    if (range[i] > value) break;
  }
  return i;
}

// match stereo hits
inline bool matchStereoHit( const int itUV1, const int uv_zone_offset_end, SciFi::HitsSoA* hits_layers, const int xMinUV, const int xMaxUV ) {

  for (int stereoHit = itUV1; stereoHit != uv_zone_offset_end; ++stereoHit) {
    if ( hits_layers->m_x[stereoHit] > xMinUV ) {
      return (hits_layers->m_x[stereoHit] < xMaxUV );
    }
  }
  return false;
}

inline bool matchStereoHitWithTriangle( const int itUV2, const int triangle_zone_offset_end, const float yInZone, SciFi::HitsSoA* hits_layers, const int xMinUV, const int xMaxUV, const int side ) {
  
  for (int stereoHit = itUV2; stereoHit != triangle_zone_offset_end; ++stereoHit) {
    if ( hits_layers->m_x[stereoHit] > xMinUV ) {
      // Triangle search condition depends on side
      if (side > 0) { // upper
        if (hits_layers->m_yMax[stereoHit] > yInZone - SciFi::Tracking::yTolUVSearch) {
          return true;
        }
      }
      else { // lower
        if (hits_layers->m_yMin[stereoHit] < yInZone + SciFi::Tracking::yTolUVSearch) {
          return true;
        }
      }
    }
  }
  return false;
}

inline void removeOutlier(
  SciFi::HitsSoA* hits_layers,
  PlaneCounter& planeCounter,
  int* coordToFit,
  int& n_coordToFit,
  const int worst ) {
  planeCounter.removeHit( hits_layers->m_planeCode[worst]/2 );
  int coordToFit_temp[SciFi::Tracking::max_coordToFit];
  int i_hit_temp = 0;
  for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit ) {
    int hit = coordToFit[i_hit];
    if (hit != worst) coordToFit_temp[i_hit_temp++] = hit;
 
  }
  n_coordToFit = i_hit_temp;
  for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit ) {
    coordToFit[i_hit] = coordToFit_temp[i_hit];
  }
  
}
