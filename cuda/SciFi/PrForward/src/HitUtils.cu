#include "HitUtils.cuh"

// match stereo hits to x hits
__host__ __device__ bool matchStereoHit( const int itUV1, const int uv_zone_offset_end, const SciFi::SciFiHits& scifi_hits, const int xMinUV, const int xMaxUV ) {

  for (int stereoHit = itUV1; stereoHit != uv_zone_offset_end; ++stereoHit) {
    if ( scifi_hits.x0[stereoHit] > xMinUV ) {
      return (scifi_hits.x0[stereoHit] < xMaxUV );
    }
  }
  return false;
}

// match stereo hits to x hits using triangle method
__host__ __device__ bool matchStereoHitWithTriangle( const int itUV2, const int triangle_zone_offset_end, const float yInZone, const SciFi::SciFiHits& scifi_hits, const int xMinUV, const int xMaxUV, const int side ) {
  
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

__host__ __device__ void removeOutlier(
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

__host__ __device__ void countPlanesOfXHits(
  PlaneCounter& planeCounter,
  const int it1,
  const int it2,
  const int n_x_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const bool usedHits[SciFi::Tracking::max_x_hits],
  const SciFi::SciFiHits& scifi_hits ) {

  planeCounter.clear();
  for (int itH = it1; itH != it2; ++itH) {
    assert( itH < n_x_hits );
    if (!usedHits[itH]) {
      const int plane = scifi_hits.planeCode[allXHits[itH]]/2;
      planeCounter.addHit( plane );
    }
  }
  
}

__host__ __device__ void countUnusedXHitsOnPlanes(
  PlaneCounter& lplaneCounter,
  const int itWindowStart,
  const int itWindowEnd,
  const int n_x_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const bool usedHits[SciFi::Tracking::max_x_hits],
  const SciFi::SciFiHits& scifi_hits){
  for (int itH = itWindowStart; itH != itWindowEnd; ++itH) {
    assert( itH < n_x_hits );
    if (!usedHits[itH]) {
      lplaneCounter.addHit( scifi_hits.planeCode[allXHits[itH]]/2 );
    }
  } 
}

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
  const SciFi::SciFiHits& scifi_hits) {

  while ( itWindowEnd <= it2 ) {
    if ( lplaneCounter.nbDifferent >= nPlanes ) {
      //have nPlanes, check x distance
      assert( itWindowEnd-1 < SciFi::Tracking::max_x_hits );
      assert( itWindowStart < SciFi::Tracking::max_x_hits );
      const float dist = coordX[itWindowEnd-1] - coordX[itWindowStart];
      if ( dist < minInterval ) {
        minInterval = dist;
        best    = itWindowStart;
        bestEnd = itWindowEnd;
      }    
    } else {
      //too few planes, add one hit
      ++itWindowEnd;
      if ( itWindowEnd > it2 ) break;
      assert( itWindowEnd <= n_x_hits );
      while( itWindowEnd<=it2  &&  usedHits[itWindowEnd-1] && itWindowEnd <= n_x_hits )
        ++itWindowEnd;
      lplaneCounter.addHit( scifi_hits.planeCode[allXHits[itWindowEnd-1]]/2 );
      continue;
    } 
    // move on to the right
    
    lplaneCounter.removeHit( scifi_hits.planeCode[allXHits[itWindowStart]]/2 );
    ++itWindowStart;
    assert( itWindowStart < itEnd );
    while( itWindowStart<itWindowEnd && usedHits[itWindowStart] && itWindowStart < n_x_hits) ++itWindowStart;
    //last hit guaranteed to be not used. Therefore there is always at least one hit to go to. No additional if required.
  }

}

__host__ __device__ void collectXHitsToFit(
  const int it1,
  const int it2,
  const int n_x_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  bool usedHits[SciFi::Tracking::max_x_hits],
  int coordToFit[SciFi::Tracking::max_x_hits],
  int& n_coordToFit,
  const float coordX[SciFi::Tracking::max_x_hits],
  float& xAtRef){

  for ( int itH = it1; it2 != itH; ++itH ) {
    assert( itH < n_x_hits );
    if (!usedHits[itH]) {
      if ( n_coordToFit >= SciFi::Tracking::max_coordToFit )
        break;
      coordToFit[n_coordToFit++] = allXHits[itH];
      usedHits[itH] = true;
      xAtRef += coordX[ itH ];
    }
  }
  xAtRef /= ((float)n_coordToFit);
}

__host__ __device__ int findBestXHitOnEmptyLayer(
  const int itEnd,
  const int itBegin,
  const SciFi::SciFiHits& scifi_hits,
  const float maxX,
  const float xPred) {
  
  float bestChi2 = 1.e9f;
  int best = -1;
  for ( int itH = itBegin ; itEnd != itH; ++itH ) {
    if( scifi_hits.x0[itH] > maxX ) break;
    const float d = scifi_hits.x0[itH] - xPred; //fast distance good enough at this point (?!)
    const float chi2 = d*d * scifi_hits.w[itH];
    if ( chi2 < bestChi2 ) {
      bestChi2 = chi2;
      best = itH;
    }    
  }
  return best;
}
                                                   
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
  int stereoHits[SciFi::Tracking::max_stereo_hits]) {
  
  for ( int itH = itBegin; itEnd != itH; ++itH ) {
    const float dx = scifi_hits.x0[itH] + yZone * scifi_hits.dxdy[itH] - xPred ;
    if ( dx >  dxTol ) break;
    if ( triangleSearch) {
      if( yZone > scifi_hits.yMax[itH] + SciFi::Tracking::yTolUVSearch)continue;
      if( yZone < scifi_hits.yMin[itH] - SciFi::Tracking::yTolUVSearch)continue;
    }
    if ( n_stereoHits >= SciFi::Tracking::max_stereo_hits )
      break;
    assert( n_stereoHits < SciFi::Tracking::max_stereo_hits );
    stereoHits[n_stereoHits] = itH;
    stereoCoords[n_stereoHits++] = dx*dxDySign;
  }
  
}

// find cluster of stereo hits with certain number of hits from different planes
// and similar dx value (stored in stereoCoords)
__host__ __device__ void findStereoHitClusterByDx(
  PlaneCounter& planeCounter,
  int& endRange,
  const SciFi::Tracking::HitSearchCuts& pars,
  float stereoCoords[SciFi::Tracking::max_stereo_hits],
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  const int n_stereoHits,
  const SciFi::SciFiHits& scifi_hits,
  float& sumCoord,
  int& first_hit) {

  while( planeCounter.nbDifferent < pars.minStereoHits ||
         stereoCoords[ endRange ] < stereoCoords[ first_hit] + SciFi::Tracking::minYGap && endRange < n_stereoHits - 1) {
    planeCounter.addHit( scifi_hits.planeCode[ stereoHits[endRange] ] / 2 );
    sumCoord += stereoCoords[ endRange ];
    ++endRange;
    if ( endRange == n_stereoHits - 1 ) break;
    first_hit = endRange-1;
  }
}

// - remove hits on planes with more than one hit if
//   the hit is farthest away from the mean
// - add hit if no hit on that plane is present in the cluster yet
//   and if the cluster spread is reduced by adding the hit
__host__ __device__ void cleanStereoHitCluster(
  int& beginRange,
  int& endRange,
  const int n_stereoHits,
  const int stereoHits[SciFi::Tracking::max_stereo_hits],
  const float stereoCoords[SciFi::Tracking::max_stereo_hits],
  float& sumCoord,
  PlaneCounter& planeCounter,
  const SciFi::SciFiHits& scifi_hits) {

  while ( endRange < n_stereoHits - 1 ) {
    const float averageCoord = sumCoord / float(endRange-beginRange);
    
    // remove first if not single and farthest from mean
    if ( planeCounter.nbInPlane( scifi_hits.planeCode[ stereoHits[beginRange] ]/2 ) > 1 &&
         ((averageCoord - stereoCoords[ beginRange ]) > 1.0f * 
          (stereoCoords[ endRange-1 ] - averageCoord)) ) {
      
      planeCounter.removeHit( scifi_hits.planeCode[ stereoHits[beginRange] ]/2 );
      sumCoord -= stereoCoords[ beginRange ];
      beginRange++;
      continue;
    }
    
    if(endRange == n_stereoHits -1 ) break; //already at end, cluster cannot be expanded anymore
    //add next, if it decreases the range size and is empty
    if ( (planeCounter.nbInPlane( scifi_hits.planeCode[ stereoHits[beginRange] ]/2 ) == 0) ) {
      if ( (averageCoord - stereoCoords[ beginRange ] > stereoCoords[ endRange ] - averageCoord ) ) {
        planeCounter.addHit( scifi_hits.planeCode[ stereoHits[endRange] ]/2 );
        sumCoord += stereoCoords[ endRange];
        endRange++;
        continue;
      }
    }
    
    break;
  }
}
 
__host__ __device__ int findBestStereoHitOnEmptyLayer(
  const int itBegin,
  const int itEnd,
  const SciFi::SciFiHits& scifi_hits,
  const float yZone,
  const float xPred,
  const float dxTol,
  const bool triangleSearch) {

  int best = -1;
  float bestChi2 = SciFi::Tracking::maxChi2Stereo;
  for (int itH = itBegin ; itEnd != itH; ++itH ) {
    const float dx = scifi_hits.x0[itH] + yZone * scifi_hits.dxdy[itH] - xPred ;
    if ( dx >  dxTol ) break;
    if ( triangleSearch) {
      if( yZone > scifi_hits.yMax[itH] + SciFi::Tracking::yTolUVSearch)continue;
      if( yZone < scifi_hits.yMin[itH] - SciFi::Tracking::yTolUVSearch)continue;
    }
    const float chi2 = dx*dx*scifi_hits.w[itH];
    if ( chi2 < bestChi2 ) {
      bestChi2 = chi2;
      best = itH;
    }    
  }   
  return best;
}
