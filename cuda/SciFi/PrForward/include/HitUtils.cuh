#pragma once

#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.cuh"

/**
   Helper functions related to properties of hits on planes
 */

// Helper used to keep track of how many x / stereo hits per lane have
// been added to a candidate track
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

__host__ __device__ void countPlanesOfXHits(
  PlaneCounter& planeCounter,
  const int it1,
  const int it2,
  const int n_x_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const bool usedHits[SciFi::Tracking::max_x_hits],
  const SciFi::SciFiHits& scifi_hits );

__host__ __device__ void countUnusedXHitsOnPlanes(
  PlaneCounter& lplaneCounter,
  const int itWindowStart,
  const int itWindowEnd,
  const int n_x_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const bool usedHits[SciFi::Tracking::max_x_hits],
  const SciFi::SciFiHits& scifi_hits);

__host__ __device__ void addXHitsForCandidateWithTooFewPlanes(
  int& itWindowStart,
  int& itWindowEnd,
  const int it2,
  const int itEnd,
  float& minInterval,
  PlaneCounter& lplaneCounter,
  const int nPlanes,
  const float coordX[SciFi::Tracking::max_x_hits],
  int& best,
  int& bestEnd,
  const bool usedHits[SciFi::Tracking::max_x_hits],
  const int n_x_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const SciFi::SciFiHits& scifi_hits);

__host__ __device__ void collectXHitsToFit(
  const int it1,
  const int it2,
  const int n_x_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  bool usedHits[SciFi::Tracking::max_x_hits],
  int coordToFit[SciFi::Tracking::max_x_hits],
  int& n_coordToFit,
  const float coordX[SciFi::Tracking::max_x_hits],
  float& xAtRef);

__host__ __device__ int findBestXHitOnEmptyLayer(
  const int itEnd,
  const int itH,
  const SciFi::SciFiHits& scifi_hits,
  const float maxX,
  const float xPred);

template<int N> __host__ __device__  void sortHitsByKey( float* keys, int n, int* hits ) {
   // find permutations
  uint permutations[N];
  assert( n <= N );
  for ( int i = 0; i < n; ++i ) {
    uint position = 0;
    for ( int j = 0; j < n; ++j ) {
      // sort keys in ascending order
      int sort_result = -1;
      if ( keys[i] > keys[j] ) sort_result = 1;
      if ( keys[i] == keys[j] ) sort_result = 0;
      position += sort_result>0 || (sort_result==0 && i>j);
    }
    permutations[position] = i;
  }

  // apply permutations, store hits in temporary container
  int hits_tmp[N];
  float keys_tmp[N];
  for ( int i = 0; i < n; ++i ) {
    const int index = permutations[i];
    hits_tmp[i] = hits[index];
    keys_tmp[i] = keys[index];
  }
  
  // copy hits back to original container
  for ( int i = 0; i < n; ++i ) {
    hits[i] = hits_tmp[i];
    keys[i] = keys_tmp[i];
  }
}

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

// match stereo hits to x hits
__host__ __device__ bool matchStereoHit(
  const int itUV1,
  const int uv_zone_offset_end,
  const SciFi::SciFiHits& scifi_hits,
  const int xMinUV,
  const int xMaxUV );

__host__ __device__ bool matchStereoHitWithTriangle(
  const int itUV2,
  const int triangle_zone_offset_end,
  const float yInZone,
  const SciFi::SciFiHits& scifi_hits,
  const int xMinUV,
  const int xMaxUV,
  const int side );

__host__ __device__ void removeOutlier(
  const SciFi::SciFiHits& scifi_hits,
  PlaneCounter& planeCounter,
  int* coordToFit,
  int& n_coordToFit,
  const int worst );

__host__ __device__ void findStereoHitsWithinXTol(
  const int itBegin,
  const int itEnd,
  const SciFi::SciFiHits& scifi_hits,
  const float yZone,
  const float xPred,
  const float dxTol,
  const bool triangleSearch,
  const float dxDySign,
  int& n_stereoHits,
  float stereoCoords[SciFi::Tracking::max_stereo_hits],
  int stereoHits[SciFi::Tracking::max_stereo_hits]);

__host__ __device__ void findStereoHitClusterByDx(
  PlaneCounter& planeCounter,
  int& endRange,
  const SciFi::Tracking::HitSearchCuts& pars,
  float stereoCoords[SciFi::Tracking::max_stereo_hits],
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  const int n_stereoHits,
  const SciFi::SciFiHits& scifi_hits,
  float& sumCoord,
  int& first_hit);

__host__ __device__ void cleanStereoHitCluster(
  int& beginRange,
  int& endRange,
  const int n_stereoHits,
  const int stereoHits[SciFi::Tracking::max_stereo_hits],
  const float stereoCoords[SciFi::Tracking::max_stereo_hits],
  float& sumCoord,
  PlaneCounter& planeCounter,
  const SciFi::SciFiHits& scifi_hits);

__host__ __device__ int findBestStereoHitOnEmptyLayer(
  const int itBegin,
  const int itEnd,
  const SciFi::SciFiHits& scifi_hits,
  const float yZone,
  const float xPred,
  const float dxTol,
  const bool triangleSearch);
