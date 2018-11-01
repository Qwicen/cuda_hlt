#include "FindStereoHits.cuh"


//=========================================================================
//  Collect all hits in the stereo planes compatible with the track
//=========================================================================
__host__ __device__ void collectStereoHits(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  SciFi::Tracking::Track& track,
  MiniState velo_state,
  SciFi::Tracking::HitSearchCuts& pars,
  SciFi::Tracking::Arrays* constArrays,
  float stereoCoords[SciFi::Tracking::max_stereo_hits],
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits)
{
  
  for ( int zone = 0; zone < SciFi::Constants::n_layers; ++zone ) {
    // get yZone and xPred: x and y values at z of layer based on candidate track parameters
    const float parsX[4] = {track.trackParams[0],
                            track.trackParams[1],
                            track.trackParams[2],
                            track.trackParams[3]};
    const float parsY[4] = {track.trackParams[4],
                            track.trackParams[5],
                            track.trackParams[6],
                            0.};
    float zZone = constArrays->uvZone_zPos[zone];
    const float yZone = evalCubicParameterization(parsY,zZone);
    assert( constArrays->uvZones[zone] < SciFi::Constants::n_zones );
    zZone += constArrays->Zone_dzdy[ constArrays->uvZones[zone] ]*yZone;  // Correct for dzDy
    const float xPred  = evalCubicParameterization(parsX,zZone);

    const bool triangleSearch = fabsf(yZone) < SciFi::Tracking::tolYTriangleSearch;
    // even zone number: if ( yZone > 0 ) continue;
    // odd zone number: if ( -yZone > 0 ) continue;
    // -> check for upper / lower half
    // -> only continue if yZone is in the correct half
    if(!triangleSearch && (2.f*float(((constArrays->uvZones[zone])%2)==0)-1.f) * yZone > 0.f) continue;

    const float dxDySign = constArrays->uvZone_dxdy[zone] < 0.f ? -1.f : 1.f;
    const float seed_x_at_zZone = xFromVelo( zZone, velo_state );
    const float dxTol = SciFi::Tracking::tolY + SciFi::Tracking::tolYSlope * (fabsf(xPred - seed_x_at_zZone) + fabsf(yZone));

    // find stereo hits whose x coordinate is within xTol of the prediction
    // from the candidate track
    // This takes the y value (max, min) into account
    const float lower_bound_at = -dxTol - yZone * constArrays->uvZone_dxdy[zone] + xPred;
    int uv_zone_offset_begin = scifi_hit_count.layer_offsets[constArrays->uvZones[zone]];
    int uv_zone_offset_end   = uv_zone_offset_begin + scifi_hit_count.n_hits_layers[constArrays->uvZones[zone]];
    
    int itBegin = getLowerBound(scifi_hits.x0, lower_bound_at, uv_zone_offset_begin, uv_zone_offset_end);
    int itEnd   = uv_zone_offset_end;

    findStereoHitsWithinXTol(
      itBegin,
      itEnd,
      scifi_hits,
      yZone,
      xPred,
      dxTol,
      triangleSearch,
      dxDySign,
      n_stereoHits,
      stereoCoords,
      stereoHits);
    
    if ( n_stereoHits >= SciFi::Tracking::max_stereo_hits )
      break;
  }

  // Sort hits by coord
  // not using thrust::sort due to temporary_buffer::allocate:: get_temporary_buffer failed" error
  //thrust::sort_by_key(thrust::seq, stereoCoords, stereoCoords + n_stereoHits, stereoHits);
  sortHitsByKey<SciFi::Tracking::max_stereo_hits>( stereoCoords, n_stereoHits, stereoHits );
}
 
//=========================================================================
//  Fit the stereo hits
//=========================================================================
__host__ __device__ bool selectStereoHits(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  SciFi::Tracking::Track& track,
  SciFi::Tracking::Arrays* constArrays,
  float stereoCoords[SciFi::Tracking::max_stereo_hits],
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  MiniState velo_state, 
  SciFi::Tracking::HitSearchCuts& pars)
{
  int bestStereoHits[SciFi::Tracking::max_stereo_hits];
  int n_bestStereoHits = 0;
  float originalYParams[3] = {track.trackParams[4],
			      track.trackParams[5],
                              track.trackParams[6]};
  float bestYParams[3];
  float bestMeanDy = 1e9f;

  if(pars.minStereoHits > n_stereoHits) return false; 
  int endLoop = n_stereoHits - pars.minStereoHits;
  
  PlaneCounter planeCounter;
  for ( int beginRange = 0; beginRange < endLoop; ++beginRange ) {
    planeCounter.clear();
    int endRange = beginRange;
    
    float sumCoord = 0.;
    // bad hack to reproduce itereator behavior from before: *(-1) = 0
    int first_hit;
    if ( endRange == 0 )
      first_hit = 0;
    else
      first_hit = endRange-1;

    findStereoHitClusterByDx(
      planeCounter,
      endRange,
      pars,
      stereoCoords,
      stereoHits,
      n_stereoHits,
      scifi_hits,
      sumCoord,
      first_hit );

    cleanStereoHitCluster(
      beginRange,
      endRange,
      n_stereoHits,
      stereoHits,
      stereoCoords,
      sumCoord,
      planeCounter,
      scifi_hits ); 
        
    //Now we have a candidate, lets fit him
    // track = original; //only yparams are changed
    track.trackParams[4] = originalYParams[0];
    track.trackParams[5] = originalYParams[1];
    track.trackParams[6] = originalYParams[2];
   
    int trackStereoHits[SciFi::Tracking::max_stereo_hits];
    int n_trackStereoHits = 0;
    assert( endRange < n_stereoHits );
    for ( int range = beginRange; range < endRange; ++range ) {
      trackStereoHits[n_trackStereoHits++] = stereoHits[range];
    }
    
    //fit Y Projection of track using stereo hits
    if(!fitYProjection(
      scifi_hits, track, trackStereoHits,
      n_trackStereoHits, planeCounter,
      velo_state, constArrays, pars)) continue;
   
    if(!addHitsOnEmptyStereoLayers(scifi_hits, scifi_hit_count, track, trackStereoHits, n_trackStereoHits, constArrays, planeCounter, velo_state, pars))continue;
    
    if(n_trackStereoHits < n_bestStereoHits) continue; //number of hits most important selection criteria!
 
    //== Calculate  dy chi2 /ndf
    float meanDy = 0.;
    assert( n_trackStereoHits < n_stereoHits );
    for ( int i_hit = 0; i_hit < n_trackStereoHits; ++i_hit ) {
      const int hit = trackStereoHits[i_hit];
      const float d = trackToHitDistance(track.trackParams, scifi_hits, hit) / scifi_hits.dxdy[hit];
      meanDy += d*d;
    }
    meanDy /=  float(n_trackStereoHits-1);

    if ( n_trackStereoHits > n_bestStereoHits || meanDy < bestMeanDy  ){
      // if same number of hits take smaller chi2
      bestYParams[0] = track.trackParams[4];
      bestYParams[1] = track.trackParams[5];
      bestYParams[2] = track.trackParams[6];
      bestMeanDy     = meanDy;

      n_bestStereoHits = 0;
      for ( int i_hit = 0; i_hit < n_trackStereoHits; ++i_hit ) {
        assert( n_bestStereoHits < SciFi::Tracking::max_stereo_hits );
        bestStereoHits[n_bestStereoHits++] = trackStereoHits[i_hit];
      }
    }

  } // beginRange loop (<endLoop)

  if ( n_bestStereoHits > 0 ) {
    track.trackParams[4] = bestYParams[0];
    track.trackParams[5] = bestYParams[1];
    track.trackParams[6] = bestYParams[2];
    assert( n_bestStereoHits < n_stereoHits );

    for ( int i_hit = 0; i_hit < n_bestStereoHits; ++i_hit ) {
      int hit = bestStereoHits[i_hit];
      if ( track.hitsNum >= SciFi::Tracking::max_scifi_hits ) break;
      assert( track.hitsNum < SciFi::Tracking::max_scifi_hits );
      track.addHit( hit );
    }
    return true;
  }
  return false;
}
 

//=========================================================================
//  Add hits on empty stereo layers, and refit if something was added
//=========================================================================
__host__ __device__ bool addHitsOnEmptyStereoLayers(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  SciFi::Tracking::Track& track,
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  SciFi::Tracking::Arrays* constArrays,
  PlaneCounter& planeCounter,
  MiniState velo_state,
  SciFi::Tracking::HitSearchCuts& pars)
{
  //at this point pc is counting only stereo HITS!
  if(planeCounter.nbDifferent  > 5) return true;

  bool added = false;
  for ( unsigned int zone = 0; zone < SciFi::Constants::n_layers; zone += 1 ) {
    assert( constArrays->uvZones[zone] < SciFi::Constants::n_zones );
    if ( planeCounter.nbInPlane( constArrays->uvZones[zone]/2 ) != 0 ) continue; //there is already one hit

    float zZone = constArrays->uvZone_zPos[zone];

    const float parsX[4] = {track.trackParams[0],
                            track.trackParams[1],
                            track.trackParams[2],
                            track.trackParams[3]};
    const float parsY[4] = {track.trackParams[4],
                            track.trackParams[5],
                            track.trackParams[6],
                            0.};

    float yZone = evalCubicParameterization(parsY,zZone);
    zZone = constArrays->Zone_dzdy[constArrays->uvZones[zone]]*yZone;  // Correct for dzDy
    yZone = evalCubicParameterization(parsY,zZone);
    const float xPred  = evalCubicParameterization(parsX,zZone);

    const bool triangleSearch = fabsf(yZone) < SciFi::Tracking::tolYTriangleSearch;
    // change sign of yZone depending on whether we are in the upper or lower half
    if(!triangleSearch && (2.f*float((((constArrays->uvZones[zone])%2)==0))-1.f) * yZone > 0.f) continue;

    //only version without triangle search!
    const float dxTol = SciFi::Tracking::tolY + SciFi::Tracking::tolYSlope * ( fabsf( xPred - velo_state.x + (zZone - velo_state.z) * velo_state.tx) + fabsf(yZone) );
    // -- Use a binary search to find the lower bound of the range of x values
    // -- This takes the y value into account
    const float lower_bound_at = -dxTol - yZone * constArrays->uvZone_dxdy[zone] + xPred;
    int uv_zone_offset_begin = scifi_hit_count.layer_offsets[constArrays->uvZones[zone]];
    int uv_zone_offset_end   = uv_zone_offset_begin + scifi_hit_count.n_hits_layers[constArrays->uvZones[zone]];
    int itBegin = getLowerBound(scifi_hits.x0,lower_bound_at,uv_zone_offset_begin,uv_zone_offset_end);
    int itEnd   = uv_zone_offset_end;
    
    int best = findBestStereoHitOnEmptyLayer(
      itBegin,
      itEnd,
      scifi_hits,
      yZone,
      xPred,
      dxTol,
      triangleSearch);
                          
    if ( -1 != best ) {
      assert( n_stereoHits < SciFi::Tracking::max_stereo_hits);
      stereoHits[n_stereoHits++] = best;
      planeCounter.addHit( scifi_hits.planeCode[best]/2 );
      added = true;
    }
  }
  if ( !added ) return true;
  return fitYProjection(
    scifi_hits, track, stereoHits,
    n_stereoHits, planeCounter,
    velo_state, constArrays, pars );
}
 
