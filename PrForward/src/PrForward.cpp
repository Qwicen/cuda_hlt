#include "PrForward.h"
//-----------------------------------------------------------------------------
// Implementation file for class : PrForward
//
// Based on code written by :
// 2012-03-20 : Olivier Callot
// 2013-03-15 : Thomas Nikodem
// 2015-02-13 : Sevda Esen [additional search in the triangles by Marian Stahl]
// 2016-03-09 : Thomas Nikodem [complete restructuring]
//-----------------------------------------------------------------------------

PrForward::PrForward() : 
  m_MLPReader_1st{mlpInputVars},
  m_MLPReader_2nd{mlpInputVars}
{}

//=============================================================================
// Main execution
//=============================================================================
std::vector<VeloUTTracking::TrackVeloUT> PrForward::operator() (
  const std::vector<VeloUTTracking::TrackVeloUT>& inputTracks,
  ForwardTracking::HitsSoAFwd *hits_layers
  ) const
{

  m_hits_layers = *hits_layers; // dereference for local member

  std::vector<VeloUTTracking::TrackVeloUT> outputTracks;
  outputTracks.reserve(inputTracks.size());

  std::cout << "About to run forward tracking for " << inputTracks.size() << " input tracks!" << std::endl;
  int numfound = 0;

  for(const VeloUTTracking::TrackVeloUT& veloTr : inputTracks) {

    std::cout << std::setprecision( 6 )
	      << "Searching for forward track for VELO-UT track with parameters " << veloTr.state_endvelo.x 
									 	  << " " 
									 	  << veloTr.state_endvelo.y
										  << " "
										  << veloTr.state_endvelo.z 
										  << " " 
										  << veloTr.state_endvelo.tx 
										  << " " 
										  << veloTr.state_endvelo.ty
										  << " " 
										  << veloTr.state_endvelo.qOverP << std::endl;

    std::vector<VeloUTTracking::TrackVeloUT> oneOutput; 
    // for a start copy input to output
    prepareOutputTrack(veloTr, oneOutput);
    numfound += oneOutput.size();
    for (auto track : oneOutput) outputTracks.emplace_back(track);
    // Reset used hits etc.
    for (int i =0; i< ForwardTracking::max_numhits_per_event; i++){
      m_hits_layers.m_used[i] = false;
      m_hits_layers.m_coord[i] = 0.0;
    }
    //break;
 
  }

  std::cout << "Found " << numfound << " forward tracks for this event!" << std::endl;

  return outputTracks;
}

//=============================================================================
void PrForward::prepareOutputTrack(
  const VeloUTTracking::TrackVeloUT& veloUTTrack,
  std::vector<VeloUTTracking::TrackVeloUT>& outputTracks
) const {

  // Cache state information from state at the end of the VELO for
  // all subsequent processing
  VeloUTTracking::FullState state_at_endvelo = veloUTTrack.state_endvelo;

  // The LHCb framework code had a PT preselection for the VeloUT tracks
  // here, which I am removing because this should be done explicitly through
  // track selectors if we do it at all, not hacked inside the tracking code

  // Some values related to the forward track which were stored in a dedicated
  // forward track class, let's see if I can get rid of that here
  const float m_zRef_track    = m_zReference; 
  const float m_xParams_seed[4] = {state_at_endvelo.x + (m_zRef_track - state_at_endvelo.z)*state_at_endvelo.tx,state_at_endvelo.tx,0.f,0.f};
  const float m_yParams_seed[4] = {state_at_endvelo.y + (m_zRef_track - state_at_endvelo.z)*state_at_endvelo.ty,state_at_endvelo.ty,0.f,0.f};

  float yAtRef = yFromVelo( m_zReference, state_at_endvelo );

  // First loop Hough cluster search, set initial search windows
  PrParameters pars_first{m_minXHits,m_maxXWindow,m_maxXWindowSlope,m_maxXGap,4u};
  PrParameters pars_second{m_minXHits_2nd,m_maxXWindow_2nd,m_maxXWindowSlope_2nd,m_maxXGap_2nd,4u};

  std::vector<int> allXHits[2];

  if(yAtRef>-5.f)collectAllXHits(allXHits[1], m_xParams_seed, m_yParams_seed, state_at_endvelo, 1); 
  if(yAtRef< 5.f)collectAllXHits(allXHits[0], m_xParams_seed, m_yParams_seed, state_at_endvelo, -1);

  std::vector<VeloUTTracking::TrackVeloUT> outputTracks1;
  
  if(yAtRef>-5.f)selectXCandidates(allXHits[1], veloUTTrack, outputTracks1, m_zRef_track, 
				   m_xParams_seed, m_yParams_seed, state_at_endvelo, pars_first,  1);
  if(yAtRef< 5.f)selectXCandidates(allXHits[0], veloUTTrack, outputTracks1, m_zRef_track, 
				   m_xParams_seed, m_yParams_seed, state_at_endvelo, pars_first, -1); 

  std::cout << "Found " << outputTracks1.size() << " X candidates in first loop" << std::endl;

  selectFullCandidates(outputTracks1,m_xParams_seed,m_yParams_seed, state_at_endvelo, pars_first);

  std::cout << "Found " << outputTracks1.size() << " full candidates in first loop" << std::endl;

  bool ok = std::any_of(outputTracks1.begin(), outputTracks1.end(),
                        [](const auto& track) {
                           return track.trackForward.hitsNum > 10;
                        });

  std::vector<VeloUTTracking::TrackVeloUT> outputTracks2; 
  if (!ok && m_secondLoop) { // If you found nothing begin the 2nd loop
    if(yAtRef>-5.f)selectXCandidates(allXHits[1], veloUTTrack, outputTracks2, m_zRef_track, 
				     m_xParams_seed, m_yParams_seed, state_at_endvelo, pars_second, 1);
    if(yAtRef< 5.f)selectXCandidates(allXHits[0], veloUTTrack, outputTracks2, m_zRef_track, 
				     m_xParams_seed, m_yParams_seed, state_at_endvelo, pars_second, -1);  

    std::cout << "Found " << outputTracks1.size() << " X candidates in second loop" << std::endl;

    selectFullCandidates(outputTracks2,m_xParams_seed,m_yParams_seed, state_at_endvelo, pars_second);

    std::cout << "Found " << outputTracks1.size() << " full candidates in second loop" << std::endl;
    // Merge
    outputTracks1.insert(std::end(outputTracks1),
		 	 std::begin(outputTracks2),
	 		 std::end(outputTracks2));
    ok = not outputTracks1.empty();
  }
 
  std::cout << "About to do final arbitration of tracks " << ok << std::endl; 
  if(ok || !m_secondLoop){
    std::sort(outputTracks1.begin(), outputTracks1.end(), lowerByQuality );
    float minQuality = m_maxQuality;
    for ( auto& track : outputTracks1 ){
      std::cout << track.trackForward.quality << " " << m_deltaQuality << " " << minQuality << std::endl;
      if(track.trackForward.quality + m_deltaQuality < minQuality) minQuality = track.trackForward.quality + m_deltaQuality;
      if(!(track.trackForward.quality > minQuality)) {
        outputTracks.emplace_back(track);
        std::cout << "Found a forward track corresponding to a velo track!" << std::endl;
      }
    }
  }
}

//=========================================================================
//  Create Full candidates out of xCandidates
//  Searching for stereo hits
//  Fit of all hits
//  save everything in track candidate folder
//=========================================================================
void PrForward::selectFullCandidates(std::vector<VeloUTTracking::TrackVeloUT>& outputTracks,
                                     const float m_xParams_seed[4],
                                     const float m_yParams_seed[4],
				     VeloUTTracking::FullState state_at_endvelo,
				     PrParameters& pars ) const {

  std::vector<unsigned int> pc;
  std::vector<float> mlpInput(7, 0.); 

  std::vector<VeloUTTracking::TrackVeloUT> selectedTracks;

  for (std::vector<VeloUTTracking::TrackVeloUT>::iterator cand = std::begin(outputTracks);
       cand != std::end(outputTracks); ++cand) {
    bool isValid = false; // In c++ this is in track class, try to understand why later
    pars.minStereoHits = 4;

    if(cand->trackForward.hitsNum + pars.minStereoHits < m_minTotalHits) {
      pars.minStereoHits = m_minTotalHits - cand->trackForward.hitsNum;
    }
    // search for hits in U/V layers
    std::vector<int> stereoHits = collectStereoHits(*cand, state_at_endvelo, pars);
    std::cout << "Collected " << stereoHits.size() << " valid stereo hits for full track search, with requirement of " << pars.minStereoHits << std::endl;
    if(stereoHits.size() < pars.minStereoHits) continue;
    // DIRTY HACK
    std::vector<std::pair<float,int> > tempforsort;
    tempforsort.clear();
    for (auto hit : stereoHits) { tempforsort.emplace_back(std::pair<float,int>(m_hits_layers.m_coord[hit],hit));}
    std::sort( tempforsort.begin(), tempforsort.end());
    stereoHits.clear();
    for (auto pair : tempforsort) {stereoHits.emplace_back(pair.second);}

    // select best U/V hits
    if ( !selectStereoHits(*cand, stereoHits, state_at_endvelo, pars) ) continue;
    std::cout << "Passed the stereo hits selection!" << std::endl;

    pc.clear();
    int planelist[12] = {0};
    // Hijacks LHCbIDs to store the values of the hits in the SoA for now, to be changed
    for (auto hit : cand->trackForward.LHCbIDs) {
      pc.push_back(hit);
      planelist[m_hits_layers.m_planeCode[hit]/2] += 1;
    }
    
    //make a fit of ALL hits
    if(!fitXProjection(cand->trackForward.trackParams, pc, planelist, pars))continue;
    std::cout << "Passed the X projection fit" << std::endl;   
 
    //check in empty x layers for hits 
    auto checked_empty = (cand->trackForward.trackParams[4]  < 0.f) ?
        addHitsOnEmptyXLayers(cand->trackForward.trackParams, m_xParams_seed, m_yParams_seed,
                              true, pc, planelist, pars, -1)
        : 
        addHitsOnEmptyXLayers(cand->trackForward.trackParams, m_xParams_seed, m_yParams_seed,
                              true, pc, planelist, pars, 1);

    if (not checked_empty) continue;
    std::cout << "Passed the empty check" << std::endl;

    //track has enough hits, calcualte quality and save if good enough
    std::cout << "Full track candidate has " << pc.size() << " hits on " << nbDifferent(planelist) << " different layers" << std::endl;    
    if(nbDifferent(planelist) >= m_minTotalHits){
      std::cout << "Computing final quality with NNs" << std::endl;

      const float qOverP  = calcqOverP(cand->trackForward.trackParams[1], state_at_endvelo);
      //orig params before fitting , TODO faster if only calc once?? mem usage?
      const float xAtRef = cand->trackForward.trackParams[0];
      float dSlope  = ( state_at_endvelo.x + (m_zReference - state_at_endvelo.z) * state_at_endvelo.tx - xAtRef ) / ( m_zReference - m_zMagnetParams[0]);
      const float zMagSlope = m_zMagnetParams[2] * pow(state_at_endvelo.tx,2) +  m_zMagnetParams[3] * pow(state_at_endvelo.ty,2);
      const float zMag    = m_zMagnetParams[0] + m_zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
      const float xMag    = state_at_endvelo.x + (zMag- state_at_endvelo.z) * state_at_endvelo.tx;
      const float slopeT  = ( xAtRef - xMag ) / ( m_zReference - zMag );
      dSlope        = slopeT - state_at_endvelo.tx;
      const float dyCoef  = dSlope * dSlope * state_at_endvelo.ty;

      float bx = slopeT;
      float ay = state_at_endvelo.y + (m_zReference - state_at_endvelo.z) * state_at_endvelo.ty;
      float by = state_at_endvelo.ty + dyCoef * m_byParams;

      //ay,by,bx params
      const float ay1  = cand->trackForward.trackParams[4];
      const float by1  = cand->trackForward.trackParams[5];
      const float bx1  = cand->trackForward.trackParams[1];

      mlpInput[0] = nbDifferent(planelist);
      mlpInput[1] = qOverP;
      mlpInput[2] = state_at_endvelo.qOverP - qOverP; //veloUT - scifi
      if(std::fabs(state_at_endvelo.qOverP) < 1e-9f) mlpInput[2] = 0.f; //no momentum estiamte
      mlpInput[3] = pow(state_at_endvelo.tx,2) + pow(state_at_endvelo.ty,2);
      mlpInput[4] = by - by1;
      mlpInput[5] = bx - bx1;
      mlpInput[6] = ay - ay1;

      float quality = 0.f;
      /// WARNING: if the NN classes straight out of TMVA are used, put a mutex here!
      if(pars.minXHits > 4) quality = m_MLPReader_1st.GetMvaValue(mlpInput); //1st loop NN
      else                   quality = m_MLPReader_2nd.GetMvaValue(mlpInput); //2nd loop NN

      quality = 1.f-quality; //backward compability

      std::cout << "Track candidate has NN quality " << quality << std::endl;

      if(quality < m_maxQuality){
	cand->trackForward.quality = quality;
	cand->trackForward.hitsNum = pc.size();
        cand->trackForward.LHCbIDs = pc;
        cand->trackForward.set_qop( qOverP );
	// Must be a neater way to do this...
	selectedTracks.emplace_back(*cand);
      }
    }
  }
  outputTracks = selectedTracks;
}

//=========================================================================
//  Fit the stereo hits
//=========================================================================
bool PrForward::selectStereoHits(VeloUTTracking::TrackVeloUT& track, 
				 std::vector<int> stereoHits,
				 VeloUTTracking::FullState state_at_endvelo, 
				 PrParameters& pars) const {
  //why do we rely on xRef? --> coord is NOT xRef for stereo HITS!
  std::vector<int> bestStereoHits;
  float originalYParams[3] = {track.trackForward.trackParams[4],
			      track.trackForward.trackParams[5],
                              track.trackForward.trackParams[6]};
  float bestYParams[3];
  float bestMeanDy       = 1e9f;

  auto beginRange = std::begin(stereoHits)-1;
  std::cout << "About to select stereo hits from list of " << stereoHits.size() << " looking for " << pars.minStereoHits << std::endl;
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
           m_hits_layers.m_coord[*endRange] < m_hits_layers.m_coord[*(endRange-1)] + m_minYGap ) {
      std::cout << "Pushing back a stereo hit " << *endRange << " on plane " << m_hits_layers.m_planeCode[*endRange] << std::endl;
      pc.push_back(*endRange);
      planelist[m_hits_layers.m_planeCode[*endRange]/2] += 1;
      sumCoord += m_hits_layers.m_coord[*endRange];
      ++endRange;
      if ( endRange == stereoHits.end() ) break;
    }

    //clean cluster
    while( true ) {
      const float averageCoord = sumCoord / float(endRange-beginRange);

      // remove first if not single and farest from mean
      if ( planelist[m_hits_layers.m_planeCode[*beginRange]/2] > 1 &&
           ((averageCoord - m_hits_layers.m_coord[*beginRange]) > 1.0f * 
	    (m_hits_layers.m_coord[*(endRange-1)] - averageCoord)) ) { //tune this value has only little effect?!
	std::cout << "Removing stereo hit " << *beginRange << " on plane " << m_hits_layers.m_planeCode[*beginRange] << std::endl;
        planelist[m_hits_layers.m_planeCode[*beginRange]/2] -= 1;
	std::vector<unsigned int> pc_temp;
        pc_temp.clear();
        for (auto hit : pc) {
          if (hit != *beginRange){
            pc_temp.push_back(hit);
          }
        }
        pc = pc_temp;
        sumCoord -= m_hits_layers.m_coord[*beginRange++];
        continue;
      }

      if(endRange == stereoHits.end()) break; //already at end, cluster cannot be expanded anymore

      //add next, if it decreases the range size and is empty
      if ( (planelist[*endRange] == 0) &&
           (averageCoord - m_hits_layers.m_coord[*beginRange] > 
	    m_hits_layers.m_coord[*endRange] - averageCoord )
         ) {
        std::cout << "Pushing back a stereo hit " << *endRange << " on plane " << m_hits_layers.m_planeCode[*endRange] << std::endl;
        pc.push_back(*endRange);
	planelist[m_hits_layers.m_planeCode[*endRange]/2] += 1;
        sumCoord += m_hits_layers.m_coord[*endRange++];
        continue;
      }

      break;
    }

    std::cout << "Found a stereo candidate, about to fit!" << std::endl;

    //Now we have a candidate, lets fit him
    // track = original; //only yparams are changed
    track.trackForward.trackParams[4] = originalYParams[0];
    track.trackForward.trackParams[5] = originalYParams[1];
    track.trackForward.trackParams[6] = originalYParams[2];
    std::vector<int> trackStereoHits;
    trackStereoHits.clear();
    // Skip a reserve from framework code 
    std::transform(beginRange, endRange, std::back_inserter(trackStereoHits),
                   [](const int& hit) { return hit; });

    //fit Y Projection of track using stereo hits
    if(!fitYProjection(track, trackStereoHits, pc, planelist, state_at_endvelo, pars))continue;
    std::cout << "Passed the Y fit" << std::endl;

    if(!addHitsOnEmptyStereoLayers(track, trackStereoHits, pc, planelist, state_at_endvelo, pars))continue;
    std::cout << "Passed adding hits on empty stereo layers" << std::endl;

    std::cout << "Selecting on size have " << trackStereoHits.size() << " want " << bestStereoHits.size() << std::endl;
    if(trackStereoHits.size() < bestStereoHits.size()) continue; //number of hits most important selection criteria!
 
    //== Calculate  dy chi2 /ndf
    float meanDy = 0.;
    for ( const auto& hit : trackStereoHits ){
      const float d = trackToHitDistance(track.trackForward.trackParams,hit) / m_hits_layers.m_dxdy[hit];
      meanDy += d*d;
    }
    meanDy /=  float(trackStereoHits.size()-1);

    if ( trackStereoHits.size() > bestStereoHits.size() || meanDy < bestMeanDy  ){
      // if same number of hits take smaller chi2
      bestYParams[0] = track.trackForward.trackParams[4];
      bestYParams[1] = track.trackForward.trackParams[5];
      bestYParams[2] = track.trackForward.trackParams[6];
      bestMeanDy     = meanDy;
      bestStereoHits = std::move(trackStereoHits);
    }

  }
  if ( bestStereoHits.size() > 0 ) {
    track.trackForward.trackParams[4] = bestYParams[0];
    track.trackForward.trackParams[5] = bestYParams[1];
    track.trackForward.trackParams[6] = bestYParams[2];
    for (auto hit : bestStereoHits) {
      track.trackForward.addLHCbID(hit);
    }
    return true;
  }
  return false;
}

//=========================================================================
//  Add hits on empty stereo layers, and refit if something was added
//=========================================================================
bool PrForward::addHitsOnEmptyStereoLayers(VeloUTTracking::TrackVeloUT& track,
                               		   std::vector<int>& stereoHits,
                               		   std::vector<unsigned int> &pc,
                               		   int planelist[],
                                     	   VeloUTTracking::FullState state_at_endvelo,
                                 	   PrParameters& pars) const {
  //at this point pc is counting only stereo HITS!
  if(nbDifferent(planelist)  > 5) return true;

  bool added = false;
  for ( unsigned int zone = 0; zone < 12; zone += 1 ) {
    if ( planelist[ m_uvZones[zone]/2 ] != 0 ) continue; //there is already one hit

    float zZone = m_uvZone_zPos[zone];

    const float parsX[4] = {track.trackForward.trackParams[0],
                            track.trackForward.trackParams[1],
                            track.trackForward.trackParams[2],
                            track.trackForward.trackParams[3]};
    const float parsY[4] = {track.trackForward.trackParams[4],
                            track.trackForward.trackParams[5],
                            track.trackForward.trackParams[6],
                            0.};

    float yZone = straightLineExtend(parsY,zZone);
    zZone = m_Zone_dzdy[m_uvZones[zone]]*yZone;  // Correct for dzDy
    yZone = straightLineExtend(parsY,zZone);
    const float xPred  = straightLineExtend(parsX,zZone);

    const bool triangleSearch = std::fabs(yZone) < m_tolYTriangleSearch;
    // Not 100% sure about logic of next line, check it!
    if(!triangleSearch && (2.f*float((((m_uvZones[zone])%2)==0))-1.f) * yZone > 0.f) continue;

    //only version without triangle search!
    const float dxTol = m_tolY + m_tolYSlope * ( fabs( xPred - state_at_endvelo.x + (zZone - state_at_endvelo.z) * state_at_endvelo.tx) + fabs(yZone) );
    // -- Use a binary search to find the lower bound of the range of x values
    // -- This takes the y value into account
    const float lower_bound_at = -dxTol - yZone * m_uvZone_dxdy[zone] + xPred;
    int uv_zone_offset_begin = m_hits_layers.layer_offset[m_uvZones[zone]];
    int uv_zone_offset_end   = m_hits_layers.layer_offset[m_uvZones[zone]+1];
    int itH   = getLowerBound(m_hits_layers.m_x,lower_bound_at,uv_zone_offset_begin,uv_zone_offset_end);
    int itEnd = uv_zone_offset_end;
    
    int best = -1;
    float bestChi2 = m_maxChi2Stereo;
    if(triangleSearch){
      for ( ; itEnd != itH; ++itH ) {
        const float dx = m_hits_layers.m_x[itH] + yZone * m_hits_layers.m_dxdy[itH] - xPred ;
        if ( dx >  dxTol ) break;
        if( yZone > m_hits_layers.m_yMax[itH] + m_yTolUVSearch)continue;
        if( yZone < m_hits_layers.m_yMin[itH] - m_yTolUVSearch)continue;
        const float chi2 = dx*dx*m_hits_layers.m_w[itH];
        if ( chi2 < bestChi2 ) {
          bestChi2 = chi2;
          best = itH;
        }    
      }    
    }else{
      //no triangle search, thus no min max check
      for ( ; itEnd != itH; ++itH ) {
        const float dx = m_hits_layers.m_x[itH] + yZone * m_hits_layers.m_dxdy[itH] - xPred ;
        if ( dx >  dxTol ) break;
        const float chi2 = dx*dx*m_hits_layers.m_w[itH];
        if ( chi2 < bestChi2 ) {
          bestChi2 = chi2;
          best = itH;
        }
      }
    }

    if ( -1 != best ) {
      stereoHits.push_back(best);
      pc.push_back(best);
      planelist[m_hits_layers.m_planeCode[best]/2] += 1;
      added = true;
    }
  }
  if ( !added ) return true;
  return fitYProjection( track, stereoHits, pc, planelist, state_at_endvelo, pars );
}
//=========================================================================
//  Fit the Y projection of a track, return OK if fit sucecssfull
//=========================================================================
bool PrForward::fitYProjection(VeloUTTracking::TrackVeloUT& track,
                               std::vector<int>& stereoHits,
			       std::vector<unsigned int> &pc,
                               int planelist[],
                               VeloUTTracking::FullState state_at_endvelo,
                               PrParameters& pars) const {
  std::cout << "About to fit a Y projection with " << pc.size() << " hits on " << nbDifferent(planelist) << " different planes looking for " << pars.minStereoHits << std::endl;
  if ( nbDifferent(planelist) < pars.minStereoHits ) return false;
  float maxChi2 = 1.e9f;
  bool parabola = false; //first linear than parabola
  //== Fit a line
  const float txs  = track.trackForward.trackParams[0]; // simplify overgeneral c++ calculation
  const float tsxz = state_at_endvelo.x + (m_zReference - state_at_endvelo.z) * state_at_endvelo.tx; 
  const float tolYMag = m_tolYMag + m_tolYMagSlope * fabs(txs-tsxz);
  const float wMag   = 1./(tolYMag * tolYMag );

  bool doFit = true;
  while ( doFit ) {
    //Use position in magnet as constrain in fit
    //although bevause wMag is quite small only little influence...
    float zMag  = zMagnet(state_at_endvelo);
    const float tys = track.trackForward.trackParams[4]+(zMag-m_zReference)*track.trackForward.trackParams[5];
    const float tsyz = state_at_endvelo.y + (zMag-state_at_endvelo.z)*state_at_endvelo.ty;
    const float dyMag = tys-tsyz;
    zMag -= m_zReference;
    float s0   = wMag;
    float sz   = wMag * zMag;
    float sz2  = wMag * zMag * zMag;
    float sd   = wMag * dyMag;
    float sdz  = wMag * dyMag * zMag;

    std::vector<int>::const_iterator itEnd = std::end(stereoHits);

    if ( parabola ) {
      float sz2m = 0.;
      float sz3  = 0.;
      float sz4  = 0.;
      float sdz2 = 0.;

      for ( const auto hit : stereoHits ){
        const float d = - trackToHitDistance(track.trackForward.trackParams,hit) / 
			  m_hits_layers.m_dxdy[hit];//TODO multiplication much faster than division!
        const float w = m_hits_layers.m_w[hit];
        const float z = m_hits_layers.m_z[hit] - m_zReference;
        s0   += w;
        sz   += w * z; 
        sz2m += w * z * z; 
        sz2  += w * z * z; 
        sz3  += w * z * z * z; 
        sz4  += w * z * z * z * z; 
        sd   += w * d; 
        sdz  += w * d * z; 
        sdz2 += w * d * z * z; 
      }    
      const float b1 = sz  * sz   - s0  * sz2; 
      const float c1 = sz2m* sz   - s0  * sz3; 
      const float d1 = sd  * sz   - s0  * sdz; 
      const float b2 = sz2 * sz2m - sz * sz3; 
      const float c2 = sz3 * sz2m - sz * sz4; 
      const float d2 = sdz * sz2m - sz * sdz2;
      const float den = (b1 * c2 - b2 * c1 );
      if(!(std::fabs(den) > 1e-5)) {
        std::cout << "Failing Y projection fit at first possible place" << std::endl;
	return false;
      }

      const float db  = (d1 * c2 - d2 * c1 ) / den; 
      const float dc  = (d2 * b1 - d1 * b2 ) / den; 
      const float da  = ( sd - db * sz - dc * sz2 ) / s0;
      track.trackForward.trackParams[4] += da;
      track.trackForward.trackParams[5] += db;
      track.trackForward.trackParams[6] += dc;
    } else {

      for ( const auto hit : stereoHits ){
        const float d = - trackToHitDistance(track.trackForward.trackParams,hit) / 
                          m_hits_layers.m_dxdy[hit];//TODO multiplication much faster than division!
        const float w = m_hits_layers.m_w[hit];
        const float z = m_hits_layers.m_z[hit] - m_zReference;
	s0   += w;
        sz   += w * z; 
        sz2  += w * z * z;
        sd   += w * d;
        sdz  += w * d * z;
      }
      const float den = (s0 * sz2 - sz * sz );
      if(!(std::fabs(den) > 1e-5)) { 
        std::cout << "Failing Y projection fit at second possible place" << std::endl;
        return false;
      }
      const float da  = (sd * sz2 - sdz * sz ) / den;
      const float db  = (sdz * s0 - sd  * sz ) / den;
      track.trackForward.trackParams[4] += da;
      track.trackForward.trackParams[5] += db;
    }//fit end, now doing outlier removal

    std::vector<int>::iterator worst = std::end(stereoHits);
    maxChi2 = 0.;
    for ( std::vector<int>::iterator itH = std::begin(stereoHits); itEnd != itH; ++itH) {
      float d = trackToHitDistance(track.trackForward.trackParams, *itH);
      float chi2 = d*d*m_hits_layers.m_w[*itH];
      if ( chi2 > maxChi2 ) {
        maxChi2 = chi2;
        worst   = itH;
      }
    }

    if ( maxChi2 < m_maxChi2StereoLinear && !parabola ) {
      std::cout << "Maximum chi2 from linear fit was relatively small " << maxChi2 << " do parabolic fit" << std::endl;
      parabola = true;
      maxChi2 = 1.e9f;
      continue;
    }

    if ( maxChi2 > m_maxChi2Stereo ) {
      std::cout << "Removing hit " << *worst << " with chi2 " << maxChi2 << " allowable was " << m_maxChi2Stereo << std::endl;
      planelist[m_hits_layers.m_planeCode[*worst]/2] -= 1;
      std::vector<unsigned int> pc_temp;
      pc_temp.clear();
      for (auto hit : pc) {
        if (hit != *worst){
          pc_temp.push_back(hit);
        }
      }
      pc = pc_temp;
      if ( nbDifferent(planelist) < pars.minStereoHits ) {
	std::cout << "Failing because we have " << nbDifferent(planelist) << " different planes and we need " << pars.minStereoHits << std::endl;
        return false;
      }
      stereoHits.erase( worst );
      continue;
    }
    break;
  }
  return true;
}
//=========================================================================
//  Collect all hits in the stereo planes compatible with the track
//=========================================================================
std::vector<int> PrForward::collectStereoHits(VeloUTTracking::TrackVeloUT& track,
                                     	      VeloUTTracking::FullState state_at_endvelo,
                                 	      PrParameters& pars) const {
  
  std::vector<int> stereoHits;
  // Skip a reserve call for now, save that for CUDA
  for ( int zone = 0; zone < 12; ++zone ) {
    float zZone = m_uvZone_zPos[zone];
    const float parsX[4] = {track.trackForward.trackParams[0],
                            track.trackForward.trackParams[1],
                            track.trackForward.trackParams[2],
                            track.trackForward.trackParams[3]};
    const float parsY[4] = {track.trackForward.trackParams[4],
                            track.trackForward.trackParams[5],
                            track.trackForward.trackParams[6],
                            0.};
    const float yZone = straightLineExtend(parsY,zZone);
    zZone += m_Zone_dzdy[m_uvZones[zone]]*yZone;  // Correct for dzDy
    const float xPred  = straightLineExtend(parsX,zZone);

    const bool triangleSearch = std::fabs(yZone) < m_tolYTriangleSearch;
    // Not 100% sure about logic of next line, check it!
    if(!triangleSearch && (2.f*float(((m_uvZones[zone])%2)==0)-1.f) * yZone > 0.f) continue;

    //float dxDySign = 1.f - 2.f *(float)(zone.dxDy()<0); // same as ? zone.dxDy()<0 : -1 : +1 , but faster??!!
    const float dxDySign = m_uvZone_dxdy[zone]<0?-1.f:1.f;
    const float seed_x_at_zZone = state_at_endvelo.x + (zZone - state_at_endvelo.z) * state_at_endvelo.tx;//Cached as we are upgrading one at a time, revisit
    const float dxTol = m_tolY + m_tolYSlope * (std::fabs(xPred - seed_x_at_zZone) + std::fabs(yZone));

    // -- Use a binary search to find the lower bound of the range of x values
    // -- This takes the y value into account
    const float lower_bound_at = -dxTol - yZone * m_uvZone_dxdy[zone] + xPred;
    int uv_zone_offset_begin = m_hits_layers.layer_offset[m_uvZones[zone]];
    int uv_zone_offset_end   = m_hits_layers.layer_offset[m_uvZones[zone]+1];
    int itH   = getLowerBound(m_hits_layers.m_x,lower_bound_at,uv_zone_offset_begin,uv_zone_offset_end);
    int itEnd = uv_zone_offset_end;

    if(triangleSearch){
      for ( ; itEnd != itH; ++itH ) {
        const float dx = m_hits_layers.m_x[itH] + yZone * m_hits_layers.m_dxdy[itH] - xPred ;
        if ( dx >  dxTol ) break;
        if( yZone > m_hits_layers.m_yMax[itH] + m_yTolUVSearch)continue;
        if( yZone < m_hits_layers.m_yMin[itH] - m_yTolUVSearch)continue;
        m_hits_layers.m_coord[itH] = dx*dxDySign;
	stereoHits.push_back(itH);
      }
    }else{ //no triangle search, thus no min max check
      for ( ; itEnd != itH; ++itH ) {
        const float dx = m_hits_layers.m_x[itH] + yZone * m_hits_layers.m_dxdy[itH] - xPred ;
        if ( dx >  dxTol ) break;
        m_hits_layers.m_coord[itH] = dx*dxDySign;
        stereoHits.push_back(itH);
      }
    }
  }
  return stereoHits;
}

//=========================================================================
// From LHCb Forward tracking description
//
// Collect all X hits, within a window defined by the minimum Pt.
// Better restrictions possible, if we use the momentum of the input track.
// Ask for the presence of a stereo hit in the same biLayer compatible.
// This reduces the efficiency. X-alone hits to be re-added later in the processing
//=========================================================================
//
void PrForward::collectAllXHits(std::vector<int>& allXHits,
				const float m_xParams_seed[4], 
				const float m_yParams_seed[4],
                                VeloUTTracking::FullState state_at_endvelo,  
				int side) const {
  // A bunch of hardcoded numbers to set the search window
  // really this should all be made configurable
  float dxRef = 0.9 * calcDxRef(m_minPt, state_at_endvelo);
  float zMag = zMagnet(state_at_endvelo);
 
  const float q = state_at_endvelo.qOverP>0.f ? 1.f :-1.f;
  const float dir = q*m_magscalefactor*(-1.f);

  // Is PT at end VELO same as PT at beamline? Check output of VeloUT
  float m_slope2 = pow(state_at_endvelo.tx,2) + pow(state_at_endvelo.ty,2); 
  const float pt = std::sqrt(std::fabs(1./(state_at_endvelo.qOverP*state_at_endvelo.qOverP))*(m_slope2)/(1.+m_slope2));
  const bool wSignTreatment = m_useWrongSignWindow && pt>m_wrongSignPT;

  float dxRefWS = 0.0; 
  if( wSignTreatment ){
    dxRefWS = 0.9 * calcDxRef(m_wrongSignPT, state_at_endvelo); //make windows a bit too small - FIXME check effect of this, seems wrong
  }

  std::array<int, 7> iZoneEnd; //6 x planes
  iZoneEnd[0] = 0; 
  int cptZone = 1; 

  int iZoneStartingPoint = side > 0 ? m_zoneoffsetpar : 0;

  //std::cout << "About to collect X hits for candidate from " << iZoneStartingPoint << std::endl;

  for(unsigned int iZone = iZoneStartingPoint; iZone < iZoneStartingPoint + m_zoneoffsetpar; iZone++) {
    //std::cout  << "Processing zone" << iZone << std::endl;
    const float zZone   = m_xZone_zPos[iZone-iZoneStartingPoint];
    const float xInZone = straightLineExtend(m_xParams_seed,zZone);
    const float yInZone = straightLineExtend(m_yParams_seed,zZone);
    // Now the code checks if the x and y are in the zone limits. I am really not sure
    // why this is done here, surely could just check if within limits for the last zone
    // in T3 and go from there? Need to think more about this.
    //
    // Here for now I assume the same min/max x and y for all stations, this again needs to
    // be read from some file blablabla although actually I suspect having some general tolerances
    // here is anyway good enough since we are doing a straight line extrapolation in the first place
    // so we are hardly looking precisely if the track could have hit this plane
    //std::cout << "Looking for hits compatible with x = " << xInZone << " and y = " << yInZone << " on side " << side << std::endl;
    if (side > 0) {
      if (!isInside(xInZone,m_xLim_Min,m_xLim_Max) || !isInside(yInZone,m_yLim_Min,m_yLim_Max)) continue;
    } else {
      if (!isInside(xInZone,m_xLim_Min,m_xLim_Max) || !isInside(yInZone,side*m_yLim_Max,side*m_yLim_Min)) continue;
    }
    //std::cout << "Found hits inside zone limits for plane " << iZone << " at " << zZone << std::endl;

    const float xTol  = ( zZone < m_zReference ) ? dxRef * zZone / m_zReference :  dxRef * (zZone - zMag) / ( m_zReference - zMag );
    float xMin        = xInZone - xTol;
    float xMax        = xInZone + xTol;

    if( m_useMomentumEstimate ) { //For VeloUT tracks, suppress check if track actually has qOverP set, get the option right!
      float xTolWS = 0.0;
      if( wSignTreatment ){
        xTolWS  = ( zZone < m_zReference ) ? dxRefWS * zZone / m_zReference :  dxRefWS * (zZone - zMag) / ( m_zReference - zMag );
      }
      if(dir > 0){
        xMin = xInZone - xTolWS;
      }else{
        xMax = xInZone + xTolWS;
      }
    }
   
    // Get the zone bounds 
    // For now we are getting these offsets "for free" from the data structure
    // Eventually we will need to do the raw bank to SoA transform ourselves of course
    int x_zone_offset_begin = m_hits_layers.layer_offset[m_xZones[iZone]];
    int x_zone_offset_end   = m_hits_layers.layer_offset[m_xZones[iZone]+1];
    int itH   = getLowerBound(m_hits_layers.m_x,xMin,x_zone_offset_begin,x_zone_offset_end); 
    int itEnd = getLowerBound(m_hits_layers.m_x,xMax,x_zone_offset_begin,x_zone_offset_end);

    //std::cout << itEnd << " " << itH << std::endl;

    // Skip making range but continue if the end is before or equal to the start
    if (!(itEnd > itH)) continue; 
 
    // Now match the stereo hits
    const float this_uv_z   = m_uvZone_zPos[iZone-iZoneStartingPoint];
    const float xInUv       = straightLineExtend(m_xParams_seed,this_uv_z);
    const float zRatio      = ( this_uv_z - zMag ) / ( zZone - zMag );
    const float dx          = yInZone * m_uvZone_dxdy[iZone-iZoneStartingPoint];
    const float xCentral    = xInZone + dx;
          float xPredUv     = xInUv + ( m_hits_layers.m_x[itH] - xInZone) * zRatio - dx;
          float maxDx       = m_tolYCollectX + ( std::fabs( m_hits_layers.m_x[itH] - xCentral ) + std::fabs( yInZone ) ) * m_tolYSlopeCollectX;
          float xMinUV      = xPredUv - maxDx;
    
    int uv_zone_offset_begin = m_hits_layers.layer_offset[m_uvZones[iZone]];
    int uv_zone_offset_end   = m_hits_layers.layer_offset[m_uvZones[iZone]+1];  
    int triangleOffset       = side > 0 ? -1 : 1;
    //std::cout<<iZone<<" " << m_uvZones[iZone] << " " << m_zoneoffsetpar << " " << triangleOffset << std::endl;
    int triangle_zone_offset_begin = m_hits_layers.layer_offset[m_uvZones[iZone + m_zoneoffsetpar*triangleOffset]];
    int triangle_zone_offset_end   = m_hits_layers.layer_offset[m_uvZones[iZone + m_zoneoffsetpar*triangleOffset]+1];
    //std::cout<<triangle_zone_offset_begin << " " << triangle_zone_offset_end << std::endl;
    int itUV1                = getLowerBound(m_hits_layers.m_x,xMinUV,uv_zone_offset_begin,uv_zone_offset_end);    
    int itUV2                = getLowerBound(m_hits_layers.m_x,xMinUV,triangle_zone_offset_begin,triangle_zone_offset_end);

    const float xPredUVProto =  xInUv - xInZone * zRatio - dx;
    const float maxDxProto   =  m_tolYCollectX + std::abs( yInZone ) * m_tolYSlopeCollectX;

    bool dotriangle = (std::fabs(yInZone) > m_tolYTriangleSearch); //cuts very slightly into distribution, 100% save cut is ~50
    for (int xHit = itH; xHit < itEnd; ++xHit) { //loop over all xHits in a layer between xMin and xMax
      const float xPredUv = xPredUVProto + m_hits_layers.m_x[xHit]* zRatio;
      const float maxDx   = maxDxProto   + std::fabs( m_hits_layers.m_x[xHit] -xCentral )* m_tolYSlopeCollectX;
      const float xMinUV  = xPredUv - maxDx;
      const float xMaxUV  = xPredUv + maxDx;

      bool foundmatch = false;
      // find matching stereo hit if possible
      for (int stereoHit = itUV1; stereoHit != uv_zone_offset_end; ++stereoHit) {
        if ( m_hits_layers.m_x[stereoHit] > xMinUV ) {
          if (m_hits_layers.m_x[stereoHit] < xMaxUV ) {
            allXHits.emplace_back(xHit);
            foundmatch = true;
            break;
          } else break;
        }
      }
      if (!foundmatch && dotriangle) { //Only do triangle if we fail to find regular match
        for (int stereoHit = itUV2; stereoHit != triangle_zone_offset_end; ++stereoHit) {
          if ( m_hits_layers.m_x[stereoHit] > xMinUV ) {
            // Triangle search condition depends on side
            if (side > 0) { // upper
              if (m_hits_layers.m_yMax[stereoHit] > yInZone - m_yTolUVSearch) {
                allXHits.emplace_back(xHit);
                break;
              } else break;
            } else { 
              if (m_hits_layers.m_yMin[stereoHit] < yInZone + m_yTolUVSearch) {
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
      xAtRef_SamePlaneHits(allXHits, m_xParams_seed, state_at_endvelo, iStart, iEnd); //calc xRef for all hits on same layer
    } 
  }
  // Drop the more sophisticated sort in the C++ code for now, not sure if this
  // is actually more efficient in CUDA. See line 577 then 1539-1552 of
  // /cvmfs/lhcb.cern.ch/lib/lhcb/REC/REC_v30r0/Pr/PrAlgorithms/src/PrForwardTool.cpp
  // DIRTY HACK FOR NOW
  std::vector<std::pair<float,int> > tempforsort; 
  tempforsort.clear();
  for (auto hit : allXHits) { tempforsort.emplace_back(std::pair<float,int>(m_hits_layers.m_coord[hit],hit));}
  std::sort( tempforsort.begin(), tempforsort.end());
  allXHits.clear();
  for (auto pair : tempforsort) {allXHits.emplace_back(pair.second);}
}

//=========================================================================
//  Select the zones in the allXHits array where we can have a track
//=========================================================================
void PrForward::selectXCandidates(std::vector<int>& allXHits,
				  const VeloUTTracking::TrackVeloUT& veloUTTrack, 
				  std::vector<VeloUTTracking::TrackVeloUT>& outputTracks,
			          const float m_zRef_track, 
                                  const float m_xParams_seed[4],
				  const float m_yParams_seed[4],
				  VeloUTTracking::FullState state_at_endvelo,
                                  PrParameters& pars,
				  int side) const {
  //std::cout << "Searching for X candidate based on " << allXHits.size() << " hits"<<std::endl;
  if ( allXHits.size() < pars.minXHits ) return;
  int itEnd = allXHits.size();//back();
  const float xStraight = straightLineExtend(m_xParams_seed,m_zReference);
  int it1 = 0;//allXHits.front();
  int it2 = it1; 
  pars.minStereoHits = 0;

  //Parameters for X-hit only fit, thus do not require stereo hits
  std::vector<int> otherHits[12];

  while( true ) {
    //find next unused Hits
    while (it1+pars.minXHits - 1 < itEnd && !isValid(allXHits[it1])) ++it1;
    it2 = it1 + pars.minXHits;
    while (it2 <= itEnd && !isValid(allXHits[it2-1])) ++it2;
    //std::cout << "Searching for X candidate with window " << it1 << " " << it2 << " " << itEnd << std::endl;
    if (it2 > itEnd) break;

    //define search window for Cluster
    //TODO better xWindow calculation?? how to tune this???
    const float xWindow = pars.maxXWindow + (std::fabs(m_hits_layers.m_coord[allXHits[it1]]) + 
                                              std::fabs(m_hits_layers.m_coord[allXHits[it1]]-xStraight)
                                             ) * pars.maxXWindowSlope;
    //If window is to small, go one step right
    if ((m_hits_layers.m_coord[allXHits[it2 - 1]] - 
         m_hits_layers.m_coord[allXHits[it1]]) > xWindow
        ) {
      ++it1;
      //std::cout << "Window is too small " << m_hits_layers.m_coord[allXHits[it2 - 1]] << " " << m_hits_layers.m_coord[allXHits[it1]] << " " << xWindow << " moving one step right" << std::endl;
      continue;
    }
    //std::cout << "Found suitable window " << m_hits_layers.m_coord[allXHits[it2 - 1]] << " " << m_hits_layers.m_coord[allXHits[it1]] << " " << xWindow << std::endl;
 
    // Cluster candidate found, now count planes
    // Skip this for now, it1 and it2 already encompass what we need at this point
    // try to get rid of this helper class (as pretty much all helper classes)
    std::vector<unsigned int> pc;
    pc.clear();
    int planelist[12] = {0};
    for (int itH = it1; itH != it2; ++itH) {
      if (isValid(allXHits[itH])) {
	//std::cout << "Pushing back valid hit " << itH << " on plane " << m_hits_layers.m_planeCode[allXHits[itH]] << std::endl;
        pc.push_back(allXHits[itH]);
        planelist[m_hits_layers.m_planeCode[allXHits[itH]]/2] += 1;
      }
    }   
    // Improve cluster (at the moment only add hits to the right)
    int itLast = it2 - 1;
    while (it2 < itEnd) {
      if (!isValid(allXHits[it2])) {
        ++it2;
        continue;
      } 
      //now  the first and last+1 hit exist and are not used!
      
      //Add next hit,
      // if there is only a small gap between the hits
      //    or insidetofill window and plane is still empty
      if ( ( m_hits_layers.m_coord[allXHits[it2]] < m_hits_layers.m_coord[allXHits[itLast]] + pars.maxXGap )
           || 
           ( (m_hits_layers.m_coord[allXHits[it2]] - m_hits_layers.m_coord[allXHits[it1]] < xWindow) && 
             (planelist[m_hits_layers.m_planeCode[allXHits[it2]]/2] == 0) 
           ) 
         ) {
	//std::cout << "Adding valid hit " << it2 << " on plane " << m_hits_layers.m_planeCode[allXHits[it2]] << std::endl;
        pc.push_back(allXHits[it2]);
        planelist[m_hits_layers.m_planeCode[allXHits[it2]]/2] += 1;
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
      //std::cout<<"Not enough different planes " << nbDifferent(planelist) << " starting again" <<std::endl;
      continue;
    }

    std::cout << "Found a partial X candidate" << std::endl;

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
    ForwardTracking::LineFitterPars lineFitParameters;
    lineFitParameters.m_z0 = m_zReference;
    float xAtRef = 0.;

    if ( nbSingle(planelist) >= m_minSingleHits && nbSingle(planelist) != nbDifferent(planelist) ) {
      //1) we have enough single planes (thus two) to make a straight line fit
      for(int i=0; i < 12; i++) otherHits[i].clear();
      //seperate single and double hits
      for(auto itH = it1; it2 > itH; ++itH ){
        if( !isValid(allXHits[itH]) ) continue;
        int planeCode = m_hits_layers.m_planeCode[allXHits[itH]]/2;
        if( planelist[planeCode] == 1 ){
          lineFitter.emplace_back(allXHits[itH]);
	  incrementLineFitParameters(lineFitParameters,allXHits[itH]);
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
          const float chi2 = getLineFitChi2(lineFitParameters, otherHits[i][hit] );
          if( chi2 < bestChi2 ){
            bestChi2 = chi2;
            best = hit; 
          }
        }
        lineFitter.emplace_back(otherHits[i][best]);
	incrementLineFitParameters(lineFitParameters,otherHits[i][best]);
        solveLineFit(lineFitParameters);
      }
      xAtRef = lineFitParameters.m_c0; //Used to be a helper function now a straight access
    } else {
      // 2) Try to find a small distance containing at least 5(4) different planes
      //    Most of the time do nothing
      const unsigned int nPlanes =  std::min(nbDifferent(planelist),int{5});
      int itWindowStart = it1; 
      int itWindowEnd   = it1 + nPlanes; //pointing at last+1
      //Hit is used, go to next unused one
      while( itWindowEnd<=it2  &&  !isValid(allXHits[itWindowEnd-1])) ++itWindowEnd;
      if( itWindowEnd > it2) continue; //start from very beginning

      float minInterval = 1.e9f;
      int best     = itWindowStart;
      int bestEnd  = itWindowEnd;

      std::vector<unsigned int> lpc;
      lpc.clear();
      int lplanelist[12] = {0};
      for (int itH = itWindowStart; itH != itWindowEnd; ++itH) {
        if (isValid(allXHits[itH])) {
          lpc.push_back(allXHits[itH]);
          lplanelist[m_hits_layers.m_planeCode[allXHits[itH]]/2] += 1;
	}
      } 
      while ( itWindowEnd <= it2 ) {
        if ( nbDifferent(lplanelist) >= nPlanes ) {
          //have nPlanes, check x distance
          const float dist = m_hits_layers.m_coord[allXHits[itWindowEnd-1]] - m_hits_layers.m_coord[allXHits[itWindowStart]];
          if ( dist < minInterval ) {
            minInterval = dist;
            best    = itWindowStart;
            bestEnd = itWindowEnd;
          }    
        } else {
          //too few planes, add one hit
          ++itWindowEnd;
          while( itWindowEnd<=it2  &&  !isValid(allXHits[itWindowEnd-1])) ++itWindowEnd;
          if( itWindowEnd > it2) break;
          lpc.push_back(allXHits[itWindowEnd-1]);
          lplanelist[m_hits_layers.m_planeCode[allXHits[itWindowEnd-1]]/2] += 1;
          continue;
        } 
        // move on to the right
        // OK this is super annoying but the way I've set it up, sans pointers, I have to now go through this crap
        // and remove this hit. Very very irritating but hey no pointers or helper class shit!
        lplanelist[m_hits_layers.m_planeCode[allXHits[itWindowStart]]/2] -= 1;
        std::vector<unsigned int> lpc_temp;
        lpc_temp.clear();
        for (auto hit : lpc) { 
          if (hit != allXHits[itWindowStart]) {
            lpc_temp.push_back(hit);
          }
        }
        lpc = lpc_temp;
        ++itWindowStart;
        while( itWindowStart<itWindowEnd && !isValid(allXHits[itWindowStart]) ) ++itWindowStart;
        //last hit guaranteed to be not used. Therefore there is always at least one hit to go to. No additional if required.
      }
      //TODO tune minInterval cut value
      if ( minInterval < 1.f ) {
        it1 = best;
        it2 = bestEnd;
      }
      //Fill coords and compute average x at reference
      for ( int itH = it1; it2 != itH; ++itH ) {
        if (isValid(allXHits[itH])) {
          coordToFit.push_back( itH );
          xAtRef += m_hits_layers.m_coord[allXHits[itH]];
        }
      }
      xAtRef /= ((float)coordToFit.size());
    }
    std::cout << "Found an X candidate" << std::endl;
    //=== We have a candidate :)
    //
    // The objective of what follows is to add the candidate to m_candidateOutputTracks if it passes some checks
    // A lot of this is duplicating code which is moved out into this planelist helper class in the framework
    // but again, since this all needs porting to CUDA anyway, let the people doing that decide how to handle it.
    pc.clear();
    int pcplanelist[12] = {0};
    for (int j=0;j<coordToFit.size();++j){
      pc.push_back(allXHits[coordToFit[j]]);
      pcplanelist[m_hits_layers.m_planeCode[allXHits[coordToFit[j]]]/2] += 1;
    }
    // Only unused(!) hits in coordToFit now
    // The objective of what follows is to add the candidate to m_candidateOutputTracks if it passes some checks
    bool ok = nbDifferent(pcplanelist) > 3;
    std::vector<float> trackParameters;
    if(ok){
      // In LHCb code this is a move operation replacing hits on the track candidate
      // Here I will work directly with the hits in coordToFit for now
      trackParameters = getTrackParameters(xAtRef, state_at_endvelo); 
      fastLinearFit(trackParameters, pc, pcplanelist,pars);
      addHitsOnEmptyXLayers(trackParameters, m_xParams_seed, m_yParams_seed,
                            false, pc, planelist, pars, side);
      
      ok = nbDifferent(pcplanelist) > 3;
    }
    //== Fit and remove hits...
    if (ok) ok = fitXProjection(trackParameters, pc, pcplanelist, pars);
    if (ok) ok = trackParameters[7]/trackParameters[8] < m_maxChi2PerDoF;
    if (ok) ok = addHitsOnEmptyXLayers(trackParameters, m_xParams_seed, m_yParams_seed,
                                       true, pc, planelist, pars, side);
    if (ok) {
      //set ModPrHits used , challenge in c++: we don't have the link any more!
      //here it is fairly trivial... :)
      //Do we really need isUsed in Forward? We can otherwise speed up the search quite a lot!
      // --> we need it for the second loop
      std::cout << "Pushed back an X candidate for stereo search!" << std::endl;
      VeloUTTracking::TrackVeloUT track = veloUTTrack;
      track.trackForward.LHCbIDs.clear();
      track.trackForward.chi2 = trackParameters[7];
      for (int k=0;k<7;++k){
        track.trackForward.trackParams.push_back(trackParameters[k]);
      }
      for (auto hit : pc){
        track.trackForward.addLHCbID(hit);
	m_hits_layers.m_used[hit] = true; //set as used in the SoA!
      }
      outputTracks.emplace_back(track);
    } 
    ++it1;   
  }
}

bool PrForward::addHitsOnEmptyXLayers(std::vector<float> &trackParameters,
                                      const float m_xParams_seed[4],
                                      const float m_yParams_seed[4],
				      bool fullFit,
                                      std::vector<unsigned int> &pc,
                                      int planelist[],
                                      PrParameters& pars,
				      int side) const {
  //is there an empty plane? otherwise skip here!
  if (nbDifferent(planelist) > 11) return true;
  bool  added = false;
  const float x1 = trackParameters[0];
  const float xStraight = straightLineExtend(m_xParams_seed,m_zReference);
  const float xWindow = pars.maxXWindow + ( fabs( x1 ) + fabs( x1 - xStraight ) ) * pars.maxXWindowSlope;

  int iZoneStartingPoint = side > 0 ? m_zoneoffsetpar : 0;

  for(unsigned int iZone = iZoneStartingPoint; iZone < iZoneStartingPoint + m_zoneoffsetpar; iZone++) {
    if (planelist[m_xZones[iZone]/2] != 0) continue;

    const float parsX[4] = {trackParameters[0],
                            trackParameters[1],
                            trackParameters[2],
                            trackParameters[3]};

    const float zZone  = m_xZone_zPos[iZone-iZoneStartingPoint];
    const float xPred  = straightLineExtend(parsX,zZone);
    const float minX = xPred - xWindow;
    const float maxX = xPred + xWindow;
    float bestChi2 = 1.e9f;
    int best = 0;

    // -- Use a search to find the lower bound of the range of x values
    int x_zone_offset_begin = m_hits_layers.layer_offset[m_xZones[iZone]];
    int x_zone_offset_end   = m_hits_layers.layer_offset[m_xZones[iZone]+1];
    int itH   = getLowerBound(m_hits_layers.m_x,minX,x_zone_offset_begin,x_zone_offset_end);
    int itEnd = m_hits_layers.layer_offset[m_xZones[iZone]+1];
    
    for ( ; itEnd != itH; ++itH ) {
      if( m_hits_layers.m_x[itH] > maxX ) break;
      const float d = m_hits_layers.m_x[itH] - xPred; //fast distance good enough at this point (?!)
      const float chi2 = d*d * m_hits_layers.m_w[itH];
      if ( chi2 < bestChi2 ) {
        bestChi2 = chi2;
        best = itH;
      }    
    }    
    if ( 0 != best ) {
      pc.push_back(best); // add the best hit here
      planelist[m_hits_layers.m_planeCode[best]/2] += 1;
      added = true;
    }    
  }
  if ( !added ) return true;
  if ( fullFit ) {
    return fitXProjection(trackParameters, pc, planelist,pars);
  }
  fastLinearFit( trackParameters, pc, planelist,pars);
  return true;
}

bool PrForward::fitXProjection(std::vector<float> &trackParameters,
                               std::vector<unsigned int> &pc,
                               int planelist[],
			       PrParameters& pars) const {

  if (nbDifferent(planelist) < pars.minXHits) return false;
  bool doFit = true;
  while ( doFit ) {
    //== Fit a cubic
    float s0   = 0.f; 
    float sz   = 0.f; 
    float sz2  = 0.f; 
    float sz3  = 0.f; 
    float sz4  = 0.f; 
    float sd   = 0.f; 
    float sdz  = 0.f; 
    float sdz2 = 0.f; 

    for (auto hit : pc ) {
      float d = trackToHitDistance(trackParameters,hit);
      float w = m_hits_layers.m_w[hit];
      float z = .001f * ( m_hits_layers.m_z[hit] - m_zReference );
      s0   += w;
      sz   += w * z; 
      sz2  += w * z * z; 
      sz3  += w * z * z * z; 
      sz4  += w * z * z * z * z; 
      sd   += w * d; 
      sdz  += w * d * z; 
      sdz2 += w * d * z * z; 
    }    
    const float b1 = sz  * sz  - s0  * sz2; 
    const float c1 = sz2 * sz  - s0  * sz3; 
    const float d1 = sd  * sz  - s0  * sdz; 
    const float b2 = sz2 * sz2 - sz * sz3; 
    const float c2 = sz3 * sz2 - sz * sz4; 
    const float d2 = sdz * sz2 - sz * sdz2;
    const float den = (b1 * c2 - b2 * c1 );
    if(!(std::fabs(den) > 1e-5)) return false;
    const float db  = (d1 * c2 - d2 * c1 ) / den; 
    const float dc  = (d2 * b1 - d1 * b2 ) / den; 
    const float da  = ( sd - db * sz - dc * sz2) / s0;
    trackParameters[0] += da;
    trackParameters[1] += db*1.e-3f;
    trackParameters[2] += dc*1.e-6f;    

    float maxChi2 = 0.f; 
    float totChi2 = 0.f;  
    //int   nDoF = -3; // fitted 3 parameters
    int  nDoF = -3;
    const bool notMultiple = nbDifferent(planelist) == pc.size();

    const auto itEnd = pc.back();
    auto worst = itEnd;
    for ( auto itH : pc ) {
      float d = trackToHitDistance(trackParameters, itH);
      float chi2 = d*d*m_hits_layers.m_w[itH];
      totChi2 += chi2;
      ++nDoF;
      if ( chi2 > maxChi2 && ( notMultiple || planelist[m_hits_layers.m_planeCode[itH]/2] > 1 ) ) {
        maxChi2 = chi2;
        worst   = itH; 
      }    
    }    
    if ( nDoF < 1 )return false;
    trackParameters[7] = totChi2;
    trackParameters[8] = (float) nDoF;

    if ( worst == itEnd ) {
      return true;
    }    
    doFit = false;
    if ( totChi2/nDoF > m_maxChi2PerDoF  ||
         maxChi2 > m_maxChi2XProjection ) {
      // Should really make this bit of code into a function... 
      planelist[m_hits_layers.m_planeCode[worst]/2] -= 1;
      std::vector<unsigned int> pc_temp;
      pc_temp.clear();
      for (auto hit : pc) {
        if (hit != worst) pc_temp.push_back(hit);
      }
      pc = pc_temp;
      if (nbDifferent(planelist) < pars.minXHits + pars.minStereoHits) return false;
      doFit = true;
    }    
  }
  return true;
}

void PrForward::fastLinearFit(std::vector<float> &trackParameters, 
			      std::vector<unsigned int> &pc,
			      int planelist[],
                              PrParameters& pars) const {
  bool fit = true;
  while (fit) {
    //== Fit a line
    float s0   = 0.;
    float sz   = 0.;
    float sz2  = 0.;
    float sd   = 0.;
    float sdz  = 0.;

    for (auto hit : pc ){
      const float parsX[4] = {trackParameters[0],
                              trackParameters[1],
                              trackParameters[2],
                              trackParameters[3]};
      const float zHit = m_hits_layers.m_z[hit];
      float track_x_at_zHit = straightLineExtend(parsX,zHit);
      const float d = m_hits_layers.m_x[hit] - track_x_at_zHit;
      const float w = m_hits_layers.m_w[hit];
      const float z = zHit - m_zReference;
      s0   += w;
      sz   += w * z; 
      sz2  += w * z * z; 
      sd   += w * d; 
      sdz  += w * d * z; 
    }    
    float den = (sz*sz-s0*sz2);
    if( !(std::fabs(den) > 1e-5))return;
    const float da  = (sdz * sz - sd * sz2) / den; 
    const float db  = (sd *  sz - s0 * sdz) / den; 
    trackParameters[0] += da;
    trackParameters[1] += db;
    fit = false;

    if ( pc.size() < pars.minXHits ) return;

    int worst = pc.back();
    float maxChi2 = 0.f; 
    const bool notMultiple = nbDifferent(planelist) == pc.size();
    //TODO how many multiple hits do we normaly have?
    //how often do we do the right thing here?
    //delete two hits at same time?
    for ( auto hit : pc) { 
      // This could certainly be wrapped in some helper function with a lot
      // of passing around or copying etc... 
      const float parsX[4] = {trackParameters[0],
                              trackParameters[1],
                              trackParameters[2],
                              trackParameters[3]};
      float track_x_at_zHit = straightLineExtend(parsX,m_hits_layers.m_z[hit]);
      float hitdist = m_hits_layers.m_x[hit] - track_x_at_zHit; 
      float chi2 = hitdist*hitdist*m_hits_layers.m_w[hit];

      if ( chi2 > maxChi2 && ( notMultiple || planelist[m_hits_layers.m_planeCode[hit]/2] > 1 ) ) {
        maxChi2 = chi2;
        worst   = hit; 
      }    
    }    
    //== Remove grossly out hit, or worst in multiple layers
    if ( maxChi2 > m_maxChi2LinearFit || ( !notMultiple && maxChi2 > 4.f ) ) {
      planelist[m_hits_layers.m_planeCode[worst]/2] -= 1;
      std::vector<unsigned int> pc_temp;
      pc_temp.clear();
      for (auto hit : pc) {
        if (hit != worst) pc_temp.push_back(hit);
      }
      pc = pc_temp; 
    }
  }
}

inline void PrForward::xAtRef_SamePlaneHits(std::vector<int>& allXHits,
				            const float m_xParams_seed[4],
                                     	    VeloUTTracking::FullState state_at_endvelo, 
					    int itH, int itEnd) const {
  //calculate xref for this plane
  //in the c++ this is vectorized, undoing because no point before CUDA (but vectorization is obvious)
  //this is quite computationally expensive mind, should take care when porting
  float zHit    = m_hits_layers.m_z[allXHits[itH]]; //all hits in same layer
  float xFromVelo_Hit = straightLineExtend(m_xParams_seed,zHit);
  float zMagSlope = m_zMagnetParams[2] * pow(state_at_endvelo.tx,2) +  m_zMagnetParams[3] * pow(state_at_endvelo.ty,2);
  float dSlopeDivPart = 1.f / ( zHit - m_zMagnetParams[0]);
  float dz      = 1.e-3f * ( zHit - m_zReference );
  
  while( itEnd>itH ){
    float xHit = m_hits_layers.m_x[allXHits[itH]];
    float dSlope  = ( xFromVelo_Hit - xHit ) * dSlopeDivPart;
    float zMag    = m_zMagnetParams[0] + m_zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
    float xMag    = xFromVelo_Hit + state_at_endvelo.tx * (zMag - zHit);
    float dxCoef  = dz * dz * ( m_xParams[0] + dz * m_xParams[1] ) * dSlope;
    float ratio   = (  m_zReference - zMag ) / ( zHit - zMag );
    m_hits_layers.m_coord[allXHits[itH]] = xMag + ratio * (xHit + dxCoef  - xMag);
    itH++;
  }
}
