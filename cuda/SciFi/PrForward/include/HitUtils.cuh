#pragma once

#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.cuh"

/**
   Helper functions related to properties of hits on planes
 */


struct PlaneCounter{
  int planeList[SciFi::Constants::n_layers] = {0};
  unsigned int nbDifferent = 0;

  __host__ __device__ inline void addHit( int plane ) {
    assert( plane < SciFi::Constants::n_layers );
    nbDifferent += (int)( (planeList[plane] += 1 ) == 1) ;
  }

  __host__ __device__ inline void removeHit( int plane ) {
    assert( plane < SciFi::Constants::n_layers );
    nbDifferent -= ((int)( (planeList[plane] -= 1 ) == 0)) ;
  }

  __host__ __device__ inline int nbInPlane( int plane ) const {
    assert( plane < SciFi::Constants::n_layers );
    return planeList[plane];
  }

  __host__ __device__ inline int nbSingle() const {
    int single = 0;
    for (int i=0; i < SciFi::Constants::n_layers; ++i) {
      single += planeList[i] == 1 ? 1 : 0;
    }
    return single;
  }
 
  __host__ __device__ inline void clear() {
    nbDifferent = 0;
    for ( int i = 0; i < SciFi::Constants::n_layers; ++i ) {
      planeList[i] = 0;
    }
  }
  
};


// check that val is within [min, max]
__host__ __device__ inline bool isInside(float val, const float min, const float max) {
  return (val > min) && (val < max) ;
}

// get lowest index where range[index] > value, within [start,end] of range 
__host__ __device__ inline int getLowerBound(float range[],float value,int start, int end) {
  int i = start;
  for (; i<end; i++) {
    if (range[i] > value) break;
  }
  return i;
}

// match stereo hits
__host__ __device__ inline bool matchStereoHit( const int itUV1, const int uv_zone_offset_end, const SciFi::SciFiHits& scifi_hits, const int xMinUV, const int xMaxUV ) {

  for (int stereoHit = itUV1; stereoHit != uv_zone_offset_end; ++stereoHit) {
    if ( scifi_hits.x0[stereoHit] > xMinUV ) {
      return (scifi_hits.x0[stereoHit] < xMaxUV );
    }
  }
  return false;
}

__host__ __device__ inline bool matchStereoHitWithTriangle( const int itUV2, const int triangle_zone_offset_end, const float yInZone, const SciFi::SciFiHits& scifi_hits, const int xMinUV, const int xMaxUV, const int side ) {
  
  for (int stereoHit = itUV2; stereoHit != triangle_zone_offset_end; ++stereoHit) {
    if ( scifi_hits.x0[stereoHit] > xMinUV ) {
      // Triangle search condition depends on side
      if (side > 0) { // upper
        if (scifi_hits.yMax[stereoHit] > yInZone - SciFi::Tracking::yTolUVSearch) {
          return true;
        }
      }
      else { // lower
        if (scifi_hits.yMin[stereoHit] < yInZone + SciFi::Tracking::yTolUVSearch) {
          return true;
        }
      }
    }
  }
  return false;
}

__host__ __device__ inline void removeOutlier(
  const SciFi::SciFiHits& scifi_hits,
  PlaneCounter& planeCounter,
  int* coordToFit,
  int& n_coordToFit,
  const int worst ) {
  planeCounter.removeHit( scifi_hits.planeCode[worst]/2 );
  int coordToFit_temp[SciFi::Tracking::max_stereo_hits];
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
