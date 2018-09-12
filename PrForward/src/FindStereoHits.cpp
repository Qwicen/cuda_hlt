#include "FindStereoHits.h"


//=========================================================================
//  Collect all hits in the stereo planes compatible with the track
//=========================================================================
std::vector<int> collectStereoHits(
  SciFi::Constants::HitsSoAFwd* hits_layers,
  SciFi::Constants::TrackForward& track,
  FullState state_at_endvelo,
  PrParameters& pars)
{
  
  std::vector<int> stereoHits;
  // Skip a reserve call for now, save that for CUDA
  for ( int zone = 0; zone < 12; ++zone ) {
    float zZone = SciFi::Tracking::uvZone_zPos[zone];
    const float parsX[4] = {track.trackParams[0],
                            track.trackParams[1],
                            track.trackParams[2],
                            track.trackParams[3]};
    const float parsY[4] = {track.trackParams[4],
                            track.trackParams[5],
                            track.trackParams[6],
                            0.};
    const float yZone = straightLineExtend(parsY,zZone);
    zZone += SciFi::Tracking::Zone_dzdy[SciFi::Tracking::uvZones[zone]]*yZone;  // Correct for dzDy
    const float xPred  = straightLineExtend(parsX,zZone);

    const bool triangleSearch = std::fabs(yZone) < SciFi::Tracking::tolYTriangleSearch;
    // Not 100% sure about logic of next line, check it!
    if(!triangleSearch && (2.f*float(((SciFi::Tracking::uvZones[zone])%2)==0)-1.f) * yZone > 0.f) continue;

    //float dxDySign = 1.f - 2.f *(float)(zone.dxDy()<0); // same as ? zone.dxDy()<0 : -1 : +1 , but faster??!!
    const float dxDySign = SciFi::Tracking::uvZone_dxdy[zone] < 0 ? -1.f : 1.f;
    const float seed_x_at_zZone = state_at_endvelo.x + (zZone - state_at_endvelo.z) * state_at_endvelo.tx;//Cached as we are upgrading one at a time, revisit
    const float dxTol = SciFi::Tracking::tolY + SciFi::Tracking::tolYSlope * (std::fabs(xPred - seed_x_at_zZone) + std::fabs(yZone));

    // -- Use a binary search to find the lower bound of the range of x values
    // -- This takes the y value into account
    const float lower_bound_at = -dxTol - yZone * SciFi::Tracking::uvZone_dxdy[zone] + xPred;
    int uv_zone_offset_begin = hits_layers->layer_offset[SciFi::Tracking::uvZones[zone]];
    int uv_zone_offset_end   = hits_layers->layer_offset[SciFi::Tracking::uvZones[zone]+1];
    int itH   = getLowerBound(hits_layers->m_x,lower_bound_at,uv_zone_offset_begin,uv_zone_offset_end);
    int itEnd = uv_zone_offset_end;

    if(triangleSearch){
      for ( ; itEnd != itH; ++itH ) {
        const float dx = hits_layers->m_x[itH] + yZone * hits_layers->m_dxdy[itH] - xPred ;
        if ( dx >  dxTol ) break;
        if( yZone > hits_layers->m_yMax[itH] + SciFi::Tracking::yTolUVSearch)continue;
        if( yZone < hits_layers->m_yMin[itH] - SciFi::Tracking::yTolUVSearch)continue;
        hits_layers->m_coord[itH] = dx*dxDySign;
	stereoHits.push_back(itH);
      }
    }else{ //no triangle search, thus no min max check
      for ( ; itEnd != itH; ++itH ) {
        const float dx = hits_layers->m_x[itH] + yZone * hits_layers->m_dxdy[itH] - xPred ;
        if ( dx >  dxTol ) break;
        hits_layers->m_coord[itH] = dx*dxDySign;
        stereoHits.push_back(itH);
      }
    }
  }
  return stereoHits;
}
 
//=========================================================================
//  Fit the stereo hits
//=========================================================================
bool selectStereoHits(
  SciFi::Constants::HitsSoAFwd* hits_layers,
  SciFi::Constants::TrackForward& track, 
  std::vector<int> stereoHits,
  FullState state_at_endvelo, 
  PrParameters& pars)
{
  //why do we rely on xRef? --> coord is NOT xRef for stereo HITS!
  std::vector<int> bestStereoHits;
  float originalYParams[3] = {track.trackParams[4],
			      track.trackParams[5],
                              track.trackParams[6]};
  float bestYParams[3];
  float bestMeanDy       = 1e9f;

  auto beginRange = std::begin(stereoHits)-1;
  debug_cout << "About to select stereo hits from list of " << stereoHits.size() << " looking for " << pars.minStereoHits << std::endl;
  if(pars.minStereoHits > stereoHits.size()) return false; //otherwise crash if minHits is too large
  auto endLoop = std::end(stereoHits) - pars.minStereoHits;
  std::vector<unsigned int> pc;
  int planelist[12] = {0};
  while ( beginRange < endLoop ) {
    ++beginRange;
    pc.clear(); // counting now stereo hits
    for (int k=0;k<12;++k){planelist[k]=0;}
    auto endRange = beginRange;
    float sumCoord = 0.;
    // BAD CODE RELIES ON FIRST CONDITION ALWAYS BEING TRUE AT START NOT TO SEGFAULT
    while( nbDifferent(planelist) < pars.minStereoHits ||
           hits_layers->m_coord[*endRange] < hits_layers->m_coord[*(endRange-1)] + SciFi::Tracking::minYGap ) {
      //debug_cout << "Pushing back a stereo hit " << *endRange << " on plane " << hits_layers->m_planeCode[*endRange] << std::endl;
      pc.push_back(*endRange);
      planelist[hits_layers->m_planeCode[*endRange]/2] += 1;
      sumCoord += hits_layers->m_coord[*endRange];
      ++endRange;
      if ( endRange == stereoHits.end() ) break;
    }

    //clean cluster
    while( true ) {
      const float averageCoord = sumCoord / float(endRange-beginRange);

      // remove first if not single and farthest from mean
      if ( planelist[hits_layers->m_planeCode[*beginRange]/2] > 1 &&
           ((averageCoord - hits_layers->m_coord[*beginRange]) > 1.0f * 
	    (hits_layers->m_coord[*(endRange-1)] - averageCoord)) ) { //tune this value has only little effect?!
	//debug_cout << "Removing stereo hit " << *beginRange << " on plane " << hits_layers->m_planeCode[*beginRange] << std::endl;
        planelist[hits_layers->m_planeCode[*beginRange]/2] -= 1;
	std::vector<unsigned int> pc_temp;
        pc_temp.clear();
        for (auto hit : pc) {
          if (hit != *beginRange){
            pc_temp.push_back(hit);
          }
        }
        pc = pc_temp;
        sumCoord -= hits_layers->m_coord[*beginRange++];
        continue;
      }

      if(endRange == stereoHits.end()) break; //already at end, cluster cannot be expanded anymore

      //add next, if it decreases the range size and is empty
      if ( (planelist[hits_layers->m_planeCode[*beginRange]/2] == 0) &&
           (averageCoord - hits_layers->m_coord[*beginRange] > 
	    hits_layers->m_coord[*endRange] - averageCoord )
         ) {
        //debug_cout << "Pushing back a stereo hit " << *endRange << " on plane " << hits_layers->m_planeCode[*endRange] << std::endl;
        pc.push_back(*endRange);
	planelist[hits_layers->m_planeCode[*endRange]/2] += 1;
        sumCoord += hits_layers->m_coord[*endRange++];
        continue;
      }

      break;
    }

    //debug_cout << "Found a stereo candidate, about to fit!" << std::endl;

    //Now we have a candidate, lets fit him
    // track = original; //only yparams are changed
    track.trackParams[4] = originalYParams[0];
    track.trackParams[5] = originalYParams[1];
    track.trackParams[6] = originalYParams[2];
    std::vector<int> trackStereoHits;
    trackStereoHits.clear();
    // Skip a reserve from framework code 
    std::transform(beginRange, endRange, std::back_inserter(trackStereoHits),
                   [](const int& hit) { return hit; });

    //fit Y Projection of track using stereo hits
    if(!fitYProjection(hits_layers, track, trackStereoHits, pc, planelist, state_at_endvelo, pars))continue;
    debug_cout << "Passed the Y fit" << std::endl;

    if(!addHitsOnEmptyStereoLayers(hits_layers, track, trackStereoHits, pc, planelist, state_at_endvelo, pars))continue;
    //debug_cout << "Passed adding hits on empty stereo layers" << std::endl;

    //debug_cout << "Selecting on size have " << trackStereoHits.size() << " want " << bestStereoHits.size() << std::endl;
    if(trackStereoHits.size() < bestStereoHits.size()) continue; //number of hits most important selection criteria!
 
    //== Calculate  dy chi2 /ndf
    float meanDy = 0.;
    for ( const auto& hit : trackStereoHits ){
      const float d = trackToHitDistance(track.trackParams, hits_layers, hit) / hits_layers->m_dxdy[hit];
      meanDy += d*d;
    }
    meanDy /=  float(trackStereoHits.size()-1);

    if ( trackStereoHits.size() > bestStereoHits.size() || meanDy < bestMeanDy  ){
      // if same number of hits take smaller chi2
      bestYParams[0] = track.trackParams[4];
      bestYParams[1] = track.trackParams[5];
      bestYParams[2] = track.trackParams[6];
      bestMeanDy     = meanDy;
      bestStereoHits = std::move(trackStereoHits);
    }

  }
  if ( bestStereoHits.size() > 0 ) {
    track.trackParams[4] = bestYParams[0];
    track.trackParams[5] = bestYParams[1];
    track.trackParams[6] = bestYParams[2];
    for (auto hit : bestStereoHits) {
      unsigned int LHCbID = hits_layers->m_LHCbID[hit]; 
      track.addLHCbID(LHCbID);
      track.hit_indices.push_back(hit);
    }
    return true;
  }
  return false;
}
 

//=========================================================================
//  Add hits on empty stereo layers, and refit if something was added
//=========================================================================
bool addHitsOnEmptyStereoLayers(
  SciFi::Constants::HitsSoAFwd* hits_layers,
  SciFi::Constants::TrackForward& track,
  std::vector<int>& stereoHits,
  std::vector<unsigned int> &pc,
  int planelist[],
  FullState state_at_endvelo,
  PrParameters& pars)
{
  //at this point pc is counting only stereo HITS!
  if(nbDifferent(planelist)  > 5) return true;

  bool added = false;
  for ( unsigned int zone = 0; zone < 12; zone += 1 ) {
    if ( planelist[ SciFi::Tracking::uvZones[zone]/2 ] != 0 ) continue; //there is already one hit

    float zZone = SciFi::Tracking::uvZone_zPos[zone];

    const float parsX[4] = {track.trackParams[0],
                            track.trackParams[1],
                            track.trackParams[2],
                            track.trackParams[3]};
    const float parsY[4] = {track.trackParams[4],
                            track.trackParams[5],
                            track.trackParams[6],
                            0.};

    float yZone = straightLineExtend(parsY,zZone);
    zZone = SciFi::Tracking::Zone_dzdy[SciFi::Tracking::uvZones[zone]]*yZone;  // Correct for dzDy
    yZone = straightLineExtend(parsY,zZone);
    const float xPred  = straightLineExtend(parsX,zZone);

    const bool triangleSearch = std::fabs(yZone) < SciFi::Tracking::tolYTriangleSearch;
    // Not 100% sure about logic of next line, check it!
    if(!triangleSearch && (2.f*float((((SciFi::Tracking::uvZones[zone])%2)==0))-1.f) * yZone > 0.f) continue;

    //only version without triangle search!
    const float dxTol = SciFi::Tracking::tolY + SciFi::Tracking::tolYSlope * ( fabs( xPred - state_at_endvelo.x + (zZone - state_at_endvelo.z) * state_at_endvelo.tx) + fabs(yZone) );
    // -- Use a binary search to find the lower bound of the range of x values
    // -- This takes the y value into account
    const float lower_bound_at = -dxTol - yZone * SciFi::Tracking::uvZone_dxdy[zone] + xPred;
    int uv_zone_offset_begin = hits_layers->layer_offset[SciFi::Tracking::uvZones[zone]];
    int uv_zone_offset_end   = hits_layers->layer_offset[SciFi::Tracking::uvZones[zone]+1];
    int itH   = getLowerBound(hits_layers->m_x,lower_bound_at,uv_zone_offset_begin,uv_zone_offset_end);
    int itEnd = uv_zone_offset_end;
    
    int best = -1;
    float bestChi2 = SciFi::Tracking::maxChi2Stereo;
    if(triangleSearch){
      for ( ; itEnd != itH; ++itH ) {
        const float dx = hits_layers->m_x[itH] + yZone * hits_layers->m_dxdy[itH] - xPred ;
        if ( dx >  dxTol ) break;
        if( yZone > hits_layers->m_yMax[itH] + SciFi::Tracking::yTolUVSearch)continue;
        if( yZone < hits_layers->m_yMin[itH] - SciFi::Tracking::yTolUVSearch)continue;
        const float chi2 = dx*dx*hits_layers->m_w[itH];
        if ( chi2 < bestChi2 ) {
          bestChi2 = chi2;
          best = itH;
        }    
      }    
    }else{
      //no triangle search, thus no min max check
      for ( ; itEnd != itH; ++itH ) {
        const float dx = hits_layers->m_x[itH] + yZone * hits_layers->m_dxdy[itH] - xPred ;
        if ( dx >  dxTol ) break;
        const float chi2 = dx*dx*hits_layers->m_w[itH];
        if ( chi2 < bestChi2 ) {
          bestChi2 = chi2;
          best = itH;
        }
      }
    }

    if ( -1 != best ) {
      stereoHits.push_back(best);
      pc.push_back(best);
      planelist[hits_layers->m_planeCode[best]/2] += 1;
      added = true;
    }
  }
  if ( !added ) return true;
  return fitYProjection( hits_layers, track, stereoHits, pc, planelist, state_at_endvelo, pars );
}
 
