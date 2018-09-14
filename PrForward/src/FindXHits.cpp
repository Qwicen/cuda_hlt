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
  std::vector<int>& allXHits,
  const float xParams_seed[4], 
  const float yParams_seed[4],
  const VeloState& velo_state,
  const float qOverP,
  int side)
{
  // A bunch of hardcoded numbers to set the search window
  // really this should all be made configurable
  float dxRef = 0.9 * calcDxRef(SciFi::Tracking::minPt, velo_state);
  float zMag = zMagnet(velo_state);
 
  const float q = qOverP > 0.f ? 1.f :-1.f;
  const float dir = q*SciFi::Tracking::magscalefactor*(-1.f);

  // Is PT at end VELO same as PT at beamline? Check output of VeloUT
  // DvB: no, but there is only a slight difference,
  // impact on efficiency is less than 1%,
  // the main difference probably comes from the intermediate computation of one more propagation
  float slope2 = pow(velo_state.tx,2) + pow(velo_state.ty,2); 
  const float pt = std::sqrt( std::fabs(1./ (pow(qOverP,2) ) ) * (slope2) / (1. + slope2) );
  const bool wSignTreatment = SciFi::Tracking::useWrongSignWindow && pt > SciFi::Tracking::wrongSignPT;

  float dxRefWS = 0.0; 
  if( wSignTreatment ){
    dxRefWS = 0.9 * calcDxRef(SciFi::Tracking::wrongSignPT, velo_state); //make windows a bit too small - FIXME check effect of this, seems wrong
  }

  std::array<int, 7> iZoneEnd; //6 x planes
  iZoneEnd[0] = 0; 
  int cptZone = 1; 

  int iZoneStartingPoint = side > 0 ? SciFi::Tracking::zoneoffsetpar : 0;

  //debug_cout << "About to collect X hits for candidate from " << iZoneStartingPoint << std::endl;

  for(unsigned int iZone = iZoneStartingPoint; iZone < iZoneStartingPoint + SciFi::Tracking::zoneoffsetpar; iZone++) {
    //debug_cout  << "Processing zone" << iZone << std::endl;
    const float zZone   = SciFi::Tracking::xZone_zPos[iZone-iZoneStartingPoint];
    const float xInZone = straightLineExtend(xParams_seed,zZone);
    const float yInZone = straightLineExtend(yParams_seed,zZone);
    //debug_cout  << "Extrapolated track to " << xInZone << " " << yInZone << std::endl;
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
    //debug_cout << "Found hits inside zone limits for plane " << iZone << " at " << zZone << std::endl;

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
  
    //debug_cout << "Collecting X hits from " << xMin << " to " << xMax << std::endl;
 
    // Get the zone bounds 
    // For now we are getting these offsets "for free" from the data structure
    // Eventually we will need to do the raw bank to SoA transform ourselves of course
    int x_zone_offset_begin = hits_layers->layer_offset[SciFi::Tracking::xZones[iZone]];
    int x_zone_offset_end   = hits_layers->layer_offset[SciFi::Tracking::xZones[iZone]+1];
    int itH   = getLowerBound(hits_layers->m_x,xMin,x_zone_offset_begin,x_zone_offset_end); 
    int itEnd = getLowerBound(hits_layers->m_x,xMax,x_zone_offset_begin,x_zone_offset_end);

    //for (int petlja = itH; petlja < itEnd; ++petlja) {
    //  debug_cout << hits_layers->m_LHCbID[petlja] << " " << hits_layers->m_x[petlja] << std::endl;
    //}

    //debug_cout << itEnd << " " << itH << std::endl;

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
    //debug_cout<<iZone<<" " << SciFi::Tracking::uvZones[iZone] << " " << SciFi::Tracking::zoneoffsetpar << " " << triangleOffset << std::endl;
    int triangle_zone_offset_begin = hits_layers->layer_offset[SciFi::Tracking::uvZones[iZone + SciFi::Tracking::zoneoffsetpar*triangleOffset]];
    int triangle_zone_offset_end   = hits_layers->layer_offset[SciFi::Tracking::uvZones[iZone + SciFi::Tracking::zoneoffsetpar*triangleOffset]+1];
    //debug_cout<<triangle_zone_offset_begin << " " << triangle_zone_offset_end << std::endl;
    int itUV1                = getLowerBound(hits_layers->m_x,xMinUV,uv_zone_offset_begin,uv_zone_offset_end);    
    int itUV2                = getLowerBound(hits_layers->m_x,xMinUV,triangle_zone_offset_begin,triangle_zone_offset_end);

    const float xPredUVProto =  xInUv - xInZone * zRatio - dx;
    const float maxDxProto   =  SciFi::Tracking::tolYCollectX + std::abs( yInZone ) * SciFi::Tracking::tolYSlopeCollectX;

    bool dotriangle = (std::fabs(yInZone) > SciFi::Tracking::tolYTriangleSearch); //cuts very slightly into distribution, 100% save cut is ~50
    for (int xHit = itH; xHit < itEnd; ++xHit) { //loop over all xHits in a layer between xMin and xMax
      const float xPredUv = xPredUVProto + hits_layers->m_x[xHit]* zRatio;
      const float maxDx   = maxDxProto   + std::fabs( hits_layers->m_x[xHit] -xCentral )* SciFi::Tracking::tolYSlopeCollectX;
      const float xMinUV  = xPredUv - maxDx;
      const float xMaxUV  = xPredUv + maxDx;

      bool foundmatch = false;
      // find matching stereo hit if possible
      for (int stereoHit = itUV1; stereoHit != uv_zone_offset_end; ++stereoHit) {
        if ( hits_layers->m_x[stereoHit] > xMinUV ) {
          if (hits_layers->m_x[stereoHit] < xMaxUV ) {
            allXHits.emplace_back(xHit);
            foundmatch = true;
            break;
          } else break;
        }
      }
      if (!foundmatch && dotriangle) { //Only do triangle if we fail to find regular match
        for (int stereoHit = itUV2; stereoHit != triangle_zone_offset_end; ++stereoHit) {
          if ( hits_layers->m_x[stereoHit] > xMinUV ) {
            // Triangle search condition depends on side
            if (side > 0) { // upper
              if (hits_layers->m_yMax[stereoHit] > yInZone - SciFi::Tracking::yTolUVSearch) {
                allXHits.emplace_back(xHit);
                break;
              } else break;
            } else { 
              if (hits_layers->m_yMin[stereoHit] < yInZone + SciFi::Tracking::yTolUVSearch) {
                allXHits.emplace_back(xHit);
                break;
              } else break;
            }
          }
        }
      }
    }
    const int iStart = iZoneEnd[cptZone-1];
    const int iEnd = allXHits.size();
    iZoneEnd[cptZone++] = iEnd;
    if( !(iStart == iEnd) ){
      xAtRef_SamePlaneHits(hits_layers, allXHits, xParams_seed, velo_state, iStart, iEnd); //calc xRef for all hits on same layer
    } 
  }
  // Drop the more sophisticated sort in the C++ code for now, not sure if this
  // is actually more efficient in CUDA. See line 577 then 1539-1552 of
  // /cvmfs/lhcb.cern.ch/lib/lhcb/REC/REC_v30r0/Pr/PrAlgorithms/src/PrForwardTool.cpp
  // DIRTY HACK FOR NOW
  std::vector<std::pair<float,int> > tempforsort; 
  tempforsort.clear();
  for (auto hit : allXHits) { tempforsort.emplace_back(std::pair<float,int>(hits_layers->m_coord[hit],hit));}
  std::sort( tempforsort.begin(), tempforsort.end());
  allXHits.clear();
  for (auto pair : tempforsort) {allXHits.emplace_back(pair.second);}
}



//=========================================================================
//  Select the zones in the allXHits array where we can have a track
//=========================================================================
void selectXCandidates(
  SciFi::HitsSoA* hits_layers,
  std::vector<int>& allXHits,
  const VeloUTTracking::TrackUT& veloUTTrack,
  std::vector<SciFi::Track>& outputTracks,
  const float zRef_track, 
  const float xParams_seed[4],
  const float yParams_seed[4],
  const VeloState& velo_state,
  SciFi::Tracking::HitSearchCuts& pars,
  int side)
{
  // debug_cout << "Searching for X candidate based on " << allXHits.size() << " hits"<<std::endl;
  // for (auto hit : allXHits) {
  //   debug_cout << hits_layers->m_LHCbID[hit] << " " << hits_layers->m_planeCode[hit] << " " << hits_layers->m_x[hit] << " " << hits_layers->m_yMin[hit] << " " << hits_layers->m_yMax[hit] << std::endl;
  // }
  if ( allXHits.size() < pars.minXHits ) return;
  int itEnd = allXHits.size();//back();
  const float xStraight = straightLineExtend(xParams_seed,SciFi::Tracking::zReference);
  int it1 = 0;//allXHits.front();
  int it2 = it1; 
  pars.minStereoHits = 0;

  //Parameters for X-hit only fit, thus do not require stereo hits
  std::vector<int> otherHits[12];

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
    // Skip this for now, it1 and it2 already encompass what we need at this point
    // try to get rid of this helper class (as pretty much all helper classes)
    std::vector<unsigned int> pc;
    pc.clear();
    int planelist[12] = {0};
    for (int itH = it1; itH != it2; ++itH) {
      if (hits_layers->isValid(allXHits[itH])) {
	//debug_cout << "Pushing back valid hit " << itH << " on plane " << hits_layers->m_planeCode[allXHits[itH]] << std::endl;
        pc.push_back(allXHits[itH]);
        planelist[hits_layers->m_planeCode[allXHits[itH]]/2] += 1;
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
      //    or insidetofill window and plane is still empty
      if ( ( hits_layers->m_coord[allXHits[it2]] < hits_layers->m_coord[allXHits[itLast]] + pars.maxXGap )
           || 
           ( (hits_layers->m_coord[allXHits[it2]] - hits_layers->m_coord[allXHits[it1]] < xWindow) && 
             (planelist[hits_layers->m_planeCode[allXHits[it2]]/2] == 0) 
           ) 
         ) {
	//debug_cout << "Adding valid hit " << it2 << " on plane " << hits_layers->m_planeCode[allXHits[it2]] << std::endl;
        pc.push_back(allXHits[it2]);
        planelist[hits_layers->m_planeCode[allXHits[it2]]/2] += 1;
        itLast = it2; 
        ++it2;
        continue;
      }   
      //Found nothing to improve
      break;
    }

    std::vector<int> coordToFit;
    coordToFit.clear();// In framework came with a reserve 16 call
    //if not enough different planes, start again from the very beginning with next right hit
    if (nbDifferent(planelist) < pars.minXHits) {
      ++it1;
      //debug_cout<<"Not enough different planes " << nbDifferent(planelist) << " starting again" <<std::endl;
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

    if ( nbSingle(planelist) >= SciFi::Tracking::minSingleHits && nbSingle(planelist) != nbDifferent(planelist) ) {
      //1) we have enough single planes (thus two) to make a straight line fit
      for(int i=0; i < 12; i++) otherHits[i].clear();
      //seperate single and double hits
      for(auto itH = it1; it2 > itH; ++itH ){
        if( !hits_layers->isValid(allXHits[itH]) ) continue;
        int planeCode = hits_layers->m_planeCode[allXHits[itH]]/2;
        if( planelist[planeCode] == 1 ){
          lineFitter.emplace_back(allXHits[itH]);
	  incrementLineFitParameters(lineFitParameters, hits_layers, allXHits[itH]);
        }else{
          otherHits[planeCode].emplace_back(allXHits[itH]);
        }
      }
      solveLineFit(lineFitParameters);
      //select best other hits (only best other hit is enough!)
      for(int i = 0; i < 12; i++){  //12 layers
        if(otherHits[i].empty()) continue;
        
        float bestChi2 = 1e9f;
      
        int best = 0;
        for( int hit = 0; hit < otherHits[i].size(); ++hit ){
          const float chi2 = getLineFitChi2(lineFitParameters, hits_layers, otherHits[i][hit] );
          if( chi2 < bestChi2 ){
            bestChi2 = chi2;
            best = hit; 
          }
        }
        lineFitter.emplace_back(otherHits[i][best]);
	incrementLineFitParameters(lineFitParameters, hits_layers, otherHits[i][best]);
        //solveLineFit(lineFitParameters);
      }
      solveLineFit(lineFitParameters);
      
      xAtRef = lineFitParameters.m_c0; //Used to be a helper function now a straight access
    } else { // start magical second part
      // 2) Try to find a small distance containing at least 5(4) different planes
      //    Most of the time do nothing
      const unsigned int nPlanes =  std::min(nbDifferent(planelist),int{5});
      int itWindowStart = it1; 
      int itWindowEnd   = it1 + nPlanes; //pointing at last+1
      //Hit is used, go to next unused one
      while( itWindowEnd<=it2  &&  !hits_layers->isValid(allXHits[itWindowEnd-1])) ++itWindowEnd;
      if( itWindowEnd > it2) continue; //start from very beginning
    
      float minInterval = 1.e9f;
      int best     = itWindowStart;
      int bestEnd  = itWindowEnd;

      std::vector<unsigned int> lpc;
      lpc.clear();
      int lplanelist[12] = {0};
      for (int itH = itWindowStart; itH != itWindowEnd; ++itH) {
        if (hits_layers->isValid(allXHits[itH])) {
          lpc.push_back(allXHits[itH]);
          lplanelist[hits_layers->m_planeCode[allXHits[itH]]/2] += 1;
	}
      } 
      while ( itWindowEnd <= it2 ) {
        if ( nbDifferent(lplanelist) >= nPlanes ) {
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
          lpc.push_back(allXHits[itWindowEnd-1]);
          lplanelist[hits_layers->m_planeCode[allXHits[itWindowEnd-1]]/2] += 1;
          continue;
        } 
        // move on to the right
        // OK this is super annoying but the way I've set it up, sans pointers, I have to now go through this crap
        // and remove this hit. Very very irritating but hey no pointers or helper class shit!
        // DvB: why do we do this?
        lplanelist[hits_layers->m_planeCode[allXHits[itWindowStart]]/2] -= 1;
        std::vector<unsigned int> lpc_temp;
        lpc_temp.clear();
        for (auto hit : lpc) { 
          if (hit != allXHits[itWindowStart]) {
            lpc_temp.push_back(hit);
          }
        }
        lpc = lpc_temp;
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
          coordToFit.push_back( itH );
          xAtRef += hits_layers->m_coord[allXHits[itH]];
        }
      }
      xAtRef /= ((float)coordToFit.size());
    } // end of magical second part
    //debug_cout << "Found an X candidate" << std::endl;
    //=== We have a candidate :)
    //
    // The objective of what follows is to add the candidate to m_candidateOutputTracks if it passes some checks
    // A lot of this is duplicating code which is moved out into this planelist helper class in the framework
    // but again, since this all needs porting to CUDA anyway, let the people doing that decide how to handle it.
    pc.clear();
    int pcplanelist[12] = {0};
    for (int j=0;j<coordToFit.size();++j){
      pc.push_back(allXHits[coordToFit[j]]);
      pcplanelist[hits_layers->m_planeCode[allXHits[coordToFit[j]]]/2] += 1;
    }
    // Only unused(!) hits in coordToFit now
    // The objective of what follows is to add the candidate to m_candidateOutputTracks if it passes some checks
    bool ok = nbDifferent(pcplanelist) > 3;
    std::vector<float> trackParameters;
    if(ok){
      // In LHCb code this is a move operation replacing hits on the track candidate
      // Here I will work directly with the hits in coordToFit for now
      // DvB: can this go wrong?
      trackParameters = getTrackParameters(xAtRef, velo_state); 
      fastLinearFit( hits_layers, trackParameters, pc, pcplanelist,pars);
      addHitsOnEmptyXLayers(hits_layers, trackParameters, xParams_seed, yParams_seed,
                            false, pc, planelist, pars, side);
      
      ok = nbDifferent(pcplanelist) > 3;
    }
    //== Fit and remove hits...
    if (ok) ok = fitXProjection(hits_layers, trackParameters, pc, pcplanelist, pars);
    if (ok) ok = trackParameters[7]/trackParameters[8] < SciFi::Tracking::maxChi2PerDoF;
    if (ok) ok = addHitsOnEmptyXLayers(hits_layers, trackParameters, xParams_seed, yParams_seed,
                                       true, pc, planelist, pars, side);
    if (ok) {
      //set ModPrHits used , challenge in c++: we don't have the link any more!
      //here it is fairly trivial... :)
      //Do we really need isUsed in Forward? We can otherwise speed up the search quite a lot!
      // --> we need it for the second loop
      //debug_cout << "Pushed back an X candidate for stereo search!" << std::endl;
      SciFi::Track track;
      track.state_endvelo = velo_state;
      //track.LHCbIDs.clear();
      track.chi2 = trackParameters[7];
      for (int k=0;k<7;++k){
        track.trackParams.push_back(trackParameters[k]);
      }
      for (auto hit : pc){
        //debug_cout << hits_layers->m_LHCbID[hit] << " " << hits_layers->m_planeCode[hit] << std::endl;
        unsigned int LHCbID = hits_layers->m_LHCbID[hit];
        track.addLHCbID(LHCbID);
        track.hit_indices.push_back(hit);
        //debug_cout << "added LHCbID to forward track with " << track.hitsNum << " hits: " << std::endl; //std::hex << track.LHCbIDs[track.hitsNum - 1] << std::endl;
	hits_layers->m_used[hit] = true; //set as used in the SoA!
      }
      outputTracks.emplace_back(track);
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
  std::vector<unsigned int> &pc,
  int planelist[],
  SciFi::Tracking::HitSearchCuts& pars,
  int side)
{
  //is there an empty plane? otherwise skip here!
  if (nbDifferent(planelist) > 11) return true;
  bool  added = false;
  const float x1 = trackParameters[0];
  const float xStraight = straightLineExtend(xParams_seed,SciFi::Tracking::zReference);
  const float xWindow = pars.maxXWindow + ( fabs( x1 ) + fabs( x1 - xStraight ) ) * pars.maxXWindowSlope;

  int iZoneStartingPoint = side > 0 ? SciFi::Tracking::zoneoffsetpar : 0;

  for(unsigned int iZone = iZoneStartingPoint; iZone < iZoneStartingPoint + SciFi::Tracking::zoneoffsetpar; iZone++) {
    if (planelist[SciFi::Tracking::xZones[iZone]/2] != 0) continue;

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
      pc.push_back(best); // add the best hit here
      planelist[hits_layers->m_planeCode[best]/2] += 1;
      added = true;
    }    
  }
  if ( !added ) return true;
  if ( fullFit ) {
    return fitXProjection(hits_layers, trackParameters, pc, planelist,pars);
  }
  fastLinearFit( hits_layers, trackParameters, pc, planelist,pars);
  return true;
}


 
