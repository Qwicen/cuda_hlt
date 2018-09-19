#include "FindXHits.h"



//=========================================================================
// From LHCb Forward tracking description
//
// Collect all X hits, within a window defined by the minimum Pt.
// Better restrictions possible, if we use the momentum of the input track.
// Ask for the presence of a stereo hit in the same biLayer compatible.
// This reduces the efficiency. X-alone hits to be re-added later in the processing
//
// side = 1  -> upper y half
// side = -1 -> lower y half
//=========================================================================
//
void collectAllXHits(
  SciFi::HitsSoA* hits_layers,
  int allXHits[SciFi::Tracking::max_x_hits],
  int& n_x_hits,
  const float xParams_seed[4], 
  const float yParams_seed[4],
  const MiniState& velo_state,
  const float qOverP,
  int side)
{
  // A bunch of hardcoded numbers to set the search window
  // really this should all be made configurable
  float dxRef = 0.9 * calcDxRef(SciFi::Tracking::minPt, velo_state);
  float zMag = zMagnet(velo_state);
 
  const float q = qOverP > 0.f ? 1.f :-1.f;
  const float dir = q*SciFi::Tracking::magscalefactor*(-1.f);

  float slope2 = pow(velo_state.tx,2) + pow(velo_state.ty,2); 
  const float pt = std::sqrt( std::fabs(1./ (pow(qOverP,2) ) ) * (slope2) / (1. + slope2) );
  const bool wSignTreatment = SciFi::Tracking::useWrongSignWindow && pt > SciFi::Tracking::wrongSignPT;

  float dxRefWS = 0.0; 
  if( wSignTreatment ){
    // DvB: what happens if we use the acual momentum from VeloUT here instead of a constant?
    dxRefWS = 0.9 * calcDxRef(SciFi::Tracking::wrongSignPT, velo_state); //make windows a bit too small - FIXME check effect of this, seems wrong
  }

  int iZoneEnd[7]; //6 x planes
  iZoneEnd[0] = 0; 
  int cptZone = 1; 

  int iZoneStartingPoint = side > 0 ? SciFi::Tracking::zoneoffsetpar : 0;

  for(unsigned int iZone = iZoneStartingPoint; iZone < iZoneStartingPoint + SciFi::Tracking::zoneoffsetpar; iZone++) {
    const float zZone   = SciFi::Tracking::xZone_zPos[iZone-iZoneStartingPoint];
    const float xInZone = straightLineExtend(xParams_seed,zZone);
    const float yInZone = straightLineExtend(yParams_seed,zZone);

    // Now the code checks if the x and y are in the zone limits. I am really not sure
    // why this is done here, surely could just check if within limits for the last zone
    // in T3 and go from there? Need to think more about this.
    //
    // Here for now I assume the same min/max x and y for all stations, this again needs to
    // be read from some file blablabla although actually I suspect having some general tolerances
    // here is anyway good enough since we are doing a straight line extrapolation in the first place
    // so we are hardly looking precisely if the track could have hit this plane
    //debug_cout << "Looking for hits compatible with x = " << xInZone << " and y = " << yInZone << " on side " << side << std::endl;
    if (side > 0) {
      if (!isInside(xInZone,SciFi::Tracking::xLim_Min,SciFi::Tracking::xLim_Max)
          || !isInside(yInZone,SciFi::Tracking::yLim_Min,SciFi::Tracking::yLim_Max)) continue;
    } else {
      if (!isInside(xInZone,SciFi::Tracking::xLim_Min,SciFi::Tracking::xLim_Max)
          || !isInside(yInZone,side*SciFi::Tracking::yLim_Max,side*SciFi::Tracking::yLim_Min)) continue;
    }

    const float xTol  = ( zZone < SciFi::Tracking::zReference ) ? dxRef * zZone / SciFi::Tracking::zReference :  dxRef * (zZone - zMag) / ( SciFi::Tracking::zReference - zMag );
    float xMin        = xInZone - xTol;
    float xMax        = xInZone + xTol;

    if( SciFi::Tracking::useMomentumEstimate ) { //For VeloUT tracks, suppress check if track actually has qOverP set, get the option right!
      float xTolWS = 0.0;
      if( wSignTreatment ){
        xTolWS  = ( zZone < SciFi::Tracking::zReference ) ? dxRefWS * zZone / SciFi::Tracking::zReference :  dxRefWS * (zZone - zMag) / ( SciFi::Tracking::zReference - zMag );
      }
      if(dir > 0){
        xMin = xInZone - xTolWS;
      }else{
        xMax = xInZone + xTolWS;
      }
    }

    // Get the zone bounds 
    int x_zone_offset_begin = hits_layers->layer_offset[SciFi::Tracking::xZones[iZone]];
    int x_zone_offset_end   = hits_layers->layer_offset[SciFi::Tracking::xZones[iZone]+1];
    int itH   = getLowerBound(hits_layers->m_x,xMin,x_zone_offset_begin,x_zone_offset_end); 
    int itEnd = getLowerBound(hits_layers->m_x,xMax,x_zone_offset_begin,x_zone_offset_end);

    // Skip making range but continue if the end is before or equal to the start
    if (!(itEnd > itH)) continue; 
 
    // Now match the stereo hits
    const float this_uv_z   = SciFi::Tracking::uvZone_zPos[iZone-iZoneStartingPoint];
    const float xInUv       = straightLineExtend(xParams_seed,this_uv_z);
    const float zRatio      = ( this_uv_z - zMag ) / ( zZone - zMag );
    const float dx          = yInZone * SciFi::Tracking::uvZone_dxdy[iZone-iZoneStartingPoint];
    const float xCentral    = xInZone + dx;
          float xPredUv     = xInUv + ( hits_layers->m_x[itH] - xInZone) * zRatio - dx;
          float maxDx       = SciFi::Tracking::tolYCollectX + ( std::fabs( hits_layers->m_x[itH] - xCentral ) + std::fabs( yInZone ) ) * SciFi::Tracking::tolYSlopeCollectX;
          float xMinUV      = xPredUv - maxDx;
    
    int uv_zone_offset_begin = hits_layers->layer_offset[SciFi::Tracking::uvZones[iZone]];
    int uv_zone_offset_end   = hits_layers->layer_offset[SciFi::Tracking::uvZones[iZone]+1];  
    int triangleOffset       = side > 0 ? -1 : 1;
    int triangle_zone_offset_begin = hits_layers->layer_offset[SciFi::Tracking::uvZones[iZone + SciFi::Tracking::zoneoffsetpar*triangleOffset]];
    int triangle_zone_offset_end   = hits_layers->layer_offset[SciFi::Tracking::uvZones[iZone + SciFi::Tracking::zoneoffsetpar*triangleOffset]+1];
    int itUV1                = getLowerBound(hits_layers->m_x,xMinUV,uv_zone_offset_begin,uv_zone_offset_end);    
    int itUV2                = getLowerBound(hits_layers->m_x,xMinUV,triangle_zone_offset_begin,triangle_zone_offset_end);

    const float xPredUVProto =  xInUv - xInZone * zRatio - dx;
    const float maxDxProto   =  SciFi::Tracking::tolYCollectX + std::fabs( yInZone ) * SciFi::Tracking::tolYSlopeCollectX;

    if ( std::fabs(yInZone) > SciFi::Tracking::tolYTriangleSearch ) { // no triangle search necessary!
      
      for (int xHit = itH; xHit < itEnd; ++xHit) { //loop over all xHits in a layer between xMin and xMax
        const float xPredUv = xPredUVProto + hits_layers->m_x[xHit]* zRatio;
        const float maxDx   = maxDxProto   + std::fabs( hits_layers->m_x[xHit] -xCentral )* SciFi::Tracking::tolYSlopeCollectX;
        const float xMinUV  = xPredUv - maxDx;
        const float xMaxUV  = xPredUv + maxDx;
        
        if ( matchStereoHit( itUV1, uv_zone_offset_end, hits_layers, xMinUV, xMaxUV) ) {
          if ( n_x_hits >= SciFi::Tracking::max_x_hits )
            break;
          allXHits[n_x_hits++] = xHit;
        }
      }
    }else { // triangle search
      for (int xHit = itH; xHit < itEnd; ++xHit) {
        const float xPredUv = xPredUVProto + hits_layers->m_x[xHit]* zRatio;
        const float maxDx   = maxDxProto   + std::fabs( hits_layers->m_x[xHit] -xCentral )* SciFi::Tracking::tolYSlopeCollectX;
        const float xMinUV  = xPredUv - maxDx;
        const float xMaxUV  = xPredUv + maxDx;

        if ( matchStereoHit( itUV1, uv_zone_offset_end, hits_layers, xMinUV, xMaxUV ) || matchStereoHitWithTriangle(itUV2, triangle_zone_offset_end, yInZone, hits_layers, xMinUV, xMaxUV, side ) ) {
          if ( n_x_hits >= SciFi::Tracking::max_x_hits )
            break;
          allXHits[n_x_hits++] = xHit;
        }
      }
    }
    
    const int iStart = iZoneEnd[cptZone-1];
    const int iEnd = n_x_hits;
    iZoneEnd[cptZone++] = iEnd;
    if( !(iStart == iEnd) ){
      xAtRef_SamePlaneHits(hits_layers, allXHits, n_x_hits, xParams_seed, velo_state, iStart, iEnd); //calc xRef for all hits on same layer
    }
    if ( n_x_hits >= SciFi::Tracking::max_x_hits )
      break;
  }

  // Sort hits by coord
  int allXHits_coords[SciFi::Tracking::max_x_hits];
  for ( int i_hit = 0; i_hit < n_x_hits; ++i_hit ) {
    allXHits_coords[i_hit] = hits_layers->m_coord[ allXHits[i_hit] ];
  }
  thrust::sort_by_key(thrust::host, allXHits_coords, allXHits_coords + n_x_hits, allXHits);

}



//=========================================================================
//  Select the zones in the allXHits array where we can have a track
//=========================================================================
void selectXCandidates(
  SciFi::HitsSoA* hits_layers,
  int allXHits[SciFi::Tracking::max_x_hits],
  int& n_x_hits,
  const VeloUTTracking::TrackUT& veloUTTrack,
  SciFi::Track candidate_tracks[SciFi::max_tracks],
  int& n_candidate_tracks,
  const float zRef_track, 
  const float xParams_seed[4],
  const float yParams_seed[4],
  const MiniState& velo_state,
  SciFi::Tracking::HitSearchCuts& pars,
  int side)
{
  if ( n_x_hits < pars.minXHits ) return;
  int itEnd = n_x_hits;
  const float xStraight = straightLineExtend(xParams_seed,SciFi::Tracking::zReference);
  int it1 = 0;
  int it2 = it1; 
  pars.minStereoHits = 0;

  PlaneCounter planeCounter;
  
  while( true ) {
    //find next unused Hits
    while (it1+pars.minXHits - 1 < itEnd && !hits_layers->isValid(allXHits[it1])) ++it1;
    it2 = it1 + pars.minXHits;
    while (it2 <= itEnd && !hits_layers->isValid(allXHits[it2-1])) ++it2;
    //debug_cout << "Searching for X candidate with window " << it1 << " " << it2 << " " << itEnd << std::endl;
    if (it2 > itEnd) break;

    //define search window for Cluster
    //TODO better xWindow calculation?? how to tune this???
    const float xWindow = pars.maxXWindow + (std::fabs(hits_layers->m_coord[allXHits[it1]]) + 
                                             std::fabs(hits_layers->m_coord[allXHits[it1]] - xStraight)
                                             ) * pars.maxXWindowSlope;
    //If window is to small, go one step right
    if ((hits_layers->m_coord[allXHits[it2 - 1]] - 
         hits_layers->m_coord[allXHits[it1]]) > xWindow
        ) {
      ++it1;
      //debug_cout << "Window is too small " << hits_layers->m_coord[allXHits[it2 - 1]] << " " << hits_layers->m_coord[allXHits[it1]] << " " << xWindow << " moving one step right" << std::endl;
      continue;
    }
    //debug_cout << "Found suitable window " << hits_layers->m_coord[allXHits[it2 - 1]] << " " << hits_layers->m_coord[allXHits[it1]] << " " << xWindow << std::endl;
 
    // Cluster candidate found, now count planes
    planeCounter.clear();
    for (int itH = it1; itH != it2; ++itH) {
      if (hits_layers->isValid(allXHits[itH])) {
        const int plane = hits_layers->m_planeCode[allXHits[itH]]/2;
        planeCounter.addHit( plane );
      }
    }   
    // Improve cluster (at the moment only add hits to the right)
    int itLast = it2 - 1;
    while (it2 < itEnd) {
      if (!hits_layers->isValid(allXHits[it2])) {
        ++it2;
        continue;
      } 
      //now  the first and last+1 hit exist and are not used!
      
      //Add next hit,
      // if there is only a small gap between the hits
      //    or inside window and plane is still empty
      if ( ( hits_layers->m_coord[allXHits[it2]] < hits_layers->m_coord[allXHits[itLast]] + pars.maxXGap )
           || 
           ( (hits_layers->m_coord[allXHits[it2]] - hits_layers->m_coord[allXHits[it1]] < xWindow) && 
             (planeCounter.nbInPlane( hits_layers->m_planeCode[allXHits[it2]]/2 )  == 0)
             ) 
         ) {
	//debug_cout << "Adding valid hit " << it2 << " on plane " << hits_layers->m_planeCode[allXHits[it2]] << std::endl;
        planeCounter.addHit( hits_layers->m_planeCode[allXHits[it2]]/2 );
        itLast = it2; 
        ++it2;
        continue;
      }   
      //Found nothing to improve
      break;
    }
    
    //if not enough different planes, start again from the very beginning with next right hit
    if (planeCounter.nbDifferent < pars.minXHits) {
      ++it1;
      //debug_cout<<"Not enough different planes " << planeCounter.nbDifferent << " starting again" <<std::endl;
      continue;
    }

    //debug_cout << "Found a partial X candidate" << std::endl;

    //====================================================================
    //  Now we have a (rather) clean candidate, do best hit selection
    //  Two possibilities:
    //  1) If there are enough planes with only one hit, do a straight
    //      line fit through these and select good matching others
    //  2) Do some magic
    //==================================================================== 
 
    // The line fitter was a standalone helper class in our framework, here I've ripped
    // its functionality out, basically have a struct which contains all the parameters
    // and some light helper math functions operating on these.
    std::vector<int> lineFitter;
    lineFitter.clear(); 
    SciFi::Tracking::LineFitterPars lineFitParameters;
    lineFitParameters.m_z0 = SciFi::Tracking::zReference;
    float xAtRef = 0.;
    const unsigned int nbSingle = planeCounter.nbSingle();
    std::vector<unsigned int> coordToFit;
    coordToFit.clear();
    
    if ( nbSingle >= SciFi::Tracking::minSingleHits && nbSingle != planeCounter.nbDifferent ) {
      //1) we have enough single planes (thus two) to make a straight line fit
      int otherHits[SciFi::Constants::n_layers][SciFi::Tracking::max_other_hits] = {0};
      int nOtherHits[SciFi::Constants::n_layers] = {0};
      
      //seperate single and double hits
      for(auto itH = it1; it2 > itH; ++itH ){
        if( !hits_layers->isValid(allXHits[itH]) ) continue;
        int planeCode = hits_layers->m_planeCode[allXHits[itH]]/2;
        if( planeCounter.nbInPlane(planeCode) == 1 ){
          lineFitter.emplace_back(allXHits[itH]);
	  incrementLineFitParameters(lineFitParameters, hits_layers, allXHits[itH]);
        }else{
          if ( nOtherHits[planeCode] < SciFi::Tracking::max_other_hits ) 
            otherHits[planeCode][ nOtherHits[planeCode]++ ] = allXHits[itH];
        }
      }
      solveLineFit(lineFitParameters);

      //select best other hits (only best other hit is enough!)
      for(int i = 0; i < SciFi::Constants::n_layers; i++){ 
        if(nOtherHits[i] == 0) continue;
        
        float bestChi2 = 1e9f;
      
        int best = otherHits[i][0];
        for( int hit = 0; hit < nOtherHits[i]; ++hit ){
          const float chi2 = getLineFitChi2(lineFitParameters, hits_layers, otherHits[i][hit] );
          if( chi2 < bestChi2 ){
            bestChi2 = chi2;
            best = hit; 
          }
        }
        lineFitter.emplace_back(otherHits[i][best]);
	incrementLineFitParameters(lineFitParameters, hits_layers, otherHits[i][best]);
      }
      solveLineFit(lineFitParameters);
      
      xAtRef = lineFitParameters.m_c0; //Used to be a helper function now a straight access
    } else { // start magical second part
      // 2) Try to find a small distance containing at least 5(4) different planes
      //    Most of the time do nothing
      const unsigned int nPlanes =  std::min(planeCounter.nbDifferent,uint{5});
      int itWindowStart = it1; 
      int itWindowEnd   = it1 + nPlanes; //pointing at last+1
      //Hit is used, go to next unused one
      while( itWindowEnd<=it2  &&  !hits_layers->isValid(allXHits[itWindowEnd-1])) ++itWindowEnd;
      if( itWindowEnd > it2) continue; //start from very beginning
    
      float minInterval = 1.e9f;
      int best     = itWindowStart;
      int bestEnd  = itWindowEnd;

      PlaneCounter lplaneCounter;
      for (int itH = itWindowStart; itH != itWindowEnd; ++itH) {
        if (hits_layers->isValid(allXHits[itH])) {
          lplaneCounter.addHit( hits_layers->m_planeCode[allXHits[itH]]/2 );
	}
      } 
      while ( itWindowEnd <= it2 ) {
        if ( lplaneCounter.nbDifferent >= nPlanes ) {
          //have nPlanes, check x distance
          const float dist = hits_layers->m_coord[allXHits[itWindowEnd-1]] - hits_layers->m_coord[allXHits[itWindowStart]];
          if ( dist < minInterval ) {
            minInterval = dist;
            best    = itWindowStart;
            bestEnd = itWindowEnd;
          }    
        } else {
          //too few planes, add one hit
          ++itWindowEnd;
          while( itWindowEnd<=it2  &&  !hits_layers->isValid(allXHits[itWindowEnd-1])) ++itWindowEnd;
          if( itWindowEnd > it2) break;
          lplaneCounter.addHit( hits_layers->m_planeCode[allXHits[itWindowEnd-1]]/2 );
          continue;
        } 
        // move on to the right
        lplaneCounter.removeHit( hits_layers->m_planeCode[allXHits[itWindowStart]]/2 );
        ++itWindowStart;
        while( itWindowStart<itWindowEnd && !hits_layers->isValid(allXHits[itWindowStart]) ) ++itWindowStart;
        //last hit guaranteed to be not used. Therefore there is always at least one hit to go to. No additional if required.
      }
      //TODO tune minInterval cut value
      if ( minInterval < 1.f ) {
        it1 = best;
        it2 = bestEnd;
      }
      //Fill coords and compute average x at reference
      for ( int itH = it1; it2 != itH; ++itH ) {
        if (hits_layers->isValid(allXHits[itH])) {
          coordToFit.push_back( allXHits[itH] );
          xAtRef += hits_layers->m_coord[ allXHits[itH] ];
        }
      }
      xAtRef /= ((float)coordToFit.size());
    } // end of magical second part
    //debug_cout << "Found an X candidate" << std::endl;
    //=== We have a candidate :)
    
    planeCounter.clear();
    for (int j=0;j<coordToFit.size();++j){
      planeCounter.addHit( hits_layers->m_planeCode[ coordToFit[j] ] / 2 );
    }
    // Only unused(!) hits in coordToFit now
 
    bool ok = planeCounter.nbDifferent > 3;
    std::vector<float> trackParameters;
    if(ok){
      trackParameters = getTrackParameters(xAtRef, velo_state); 
      fastLinearFit( hits_layers, trackParameters, coordToFit, planeCounter,pars);
      addHitsOnEmptyXLayers(hits_layers, trackParameters, xParams_seed, yParams_seed,
                              false, coordToFit, planeCounter, pars, side);
      
      ok = planeCounter.nbDifferent > 3;
    }
    //== Fit and remove hits...
    if (ok) ok = fitXProjection(hits_layers, trackParameters, coordToFit, planeCounter, pars);
    if (ok) ok = trackParameters[7]/trackParameters[8] < SciFi::Tracking::maxChi2PerDoF;
    if (ok )
      ok = addHitsOnEmptyXLayers(hits_layers, trackParameters, xParams_seed, yParams_seed,
                                       true, coordToFit, planeCounter, pars, side);
    if (ok) {
      //set ModPrHits used , challenge in c++: we don't have the link any more!
      //here it is fairly trivial... :)
      //Do we really need isUsed in Forward? We can otherwise speed up the search quite a lot!
      // --> we need it for the second loop
      //debug_cout << "Pushed back an X candidate for stereo search!" << std::endl;
      SciFi::Track track;
      track.state_endvelo.x = velo_state.x;
      track.state_endvelo.y = velo_state.y;
      track.state_endvelo.z = velo_state.z;
      track.state_endvelo.tx = velo_state.tx;
      track.state_endvelo.ty = velo_state.ty;
      track.chi2 = trackParameters[7];
      for (int k=0;k<7;++k){
        track.trackParams.push_back(trackParameters[k]);
      }
      for (auto hit : coordToFit){
        //debug_cout << hits_layers->m_LHCbID[hit] << " " << hits_layers->m_planeCode[hit] << std::endl;
        unsigned int LHCbID = hits_layers->m_LHCbID[hit];
        track.addLHCbID(LHCbID);
        track.hit_indices.push_back(hit);
        //debug_cout << "added LHCbID to forward track with " << track.hitsNum << " hits: " << std::endl; //std::hex << track.LHCbIDs[track.hitsNum - 1] << std::endl;
	hits_layers->m_used[hit] = true; //set as used in the SoA!
      }
      candidate_tracks[n_candidate_tracks++] = track;
    } 
    ++it1;   
  }
} 


bool addHitsOnEmptyXLayers(
  SciFi::HitsSoA* hits_layers,
  std::vector<float> &trackParameters,
  const float xParams_seed[4],
  const float yParams_seed[4],
  bool fullFit,
  std::vector<unsigned int> &coordToFit,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars,
  int side)
{
  //is there an empty plane? otherwise skip here!
  if (planeCounter.nbDifferent > 11) return true;
  bool  added = false;
  const float x1 = trackParameters[0];
  const float xStraight = straightLineExtend(xParams_seed,SciFi::Tracking::zReference);
  const float xWindow = pars.maxXWindow + ( fabs( x1 ) + fabs( x1 - xStraight ) ) * pars.maxXWindowSlope;

  int iZoneStartingPoint = side > 0 ? SciFi::Tracking::zoneoffsetpar : 0;

  for(unsigned int iZone = iZoneStartingPoint; iZone < iZoneStartingPoint + SciFi::Tracking::zoneoffsetpar; iZone++) {
    if (planeCounter.nbInPlane( SciFi::Tracking::xZones[iZone]/2 ) != 0) continue;

    const float parsX[4] = {trackParameters[0],
                            trackParameters[1],
                            trackParameters[2],
                            trackParameters[3]};

    const float zZone  = SciFi::Tracking::xZone_zPos[iZone-iZoneStartingPoint];
    const float xPred  = straightLineExtend(parsX,zZone);
    const float minX = xPred - xWindow;
    const float maxX = xPred + xWindow;
    float bestChi2 = 1.e9f;
    int best = -1;

    // -- Use a search to find the lower bound of the range of x values
    int x_zone_offset_begin = hits_layers->layer_offset[SciFi::Tracking::xZones[iZone]];
    int x_zone_offset_end   = hits_layers->layer_offset[SciFi::Tracking::xZones[iZone]+1];
    int itH   = getLowerBound(hits_layers->m_x,minX,x_zone_offset_begin,x_zone_offset_end);
    int itEnd = x_zone_offset_end;
    
    for ( ; itEnd != itH; ++itH ) {
      if( hits_layers->m_x[itH] > maxX ) break;
      const float d = hits_layers->m_x[itH] - xPred; //fast distance good enough at this point (?!)
      const float chi2 = d*d * hits_layers->m_w[itH];
      if ( chi2 < bestChi2 ) {
        bestChi2 = chi2;
        best = itH;
      }    
    }    
    if ( best != -1 ) {
      coordToFit.push_back(best); // add the best hit here
      planeCounter.addHit( hits_layers->m_planeCode[best]/2 );
      added = true;
    }    
  }
  if ( !added ) return true;
  if ( fullFit ) {
    return fitXProjection(hits_layers, trackParameters, coordToFit, planeCounter, pars);
  }
  fastLinearFit( hits_layers, trackParameters, coordToFit, planeCounter, pars);
  return true;
}


 
