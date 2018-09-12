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
std::vector<ForwardTracking::TrackForward> PrForward::operator() (
  const std::vector<VeloUTTracking::TrackVeloUT>& inputTracks,
  ForwardTracking::HitsSoAFwd *hits_layers
  ) const
{

  m_hits_layers = *hits_layers; // dereference for local member

  std::vector<ForwardTracking::TrackForward> outputTracks;
  outputTracks.reserve(inputTracks.size());

  //  debug_cout << "About to run forward tracking for " << inputTracks.size() << " input tracks!" << std::endl;
  int numfound = 0;

  for(const VeloUTTracking::TrackVeloUT& veloTr : inputTracks) {

    std::vector<ForwardTracking::TrackForward> oneOutput; 
    find_forward_tracks(veloTr, oneOutput);
    numfound += oneOutput.size();
    for (auto track : oneOutput) {
      outputTracks.emplace_back(track);
    }
    // Reset used hits etc.
    // these should not be part of the HitsSoA struct
    for (int i =0; i< ForwardTracking::max_numhits_per_event; i++){
      m_hits_layers.m_used[i] = false;
      m_hits_layers.m_coord[i] = 0.0;
    }
  }

  //  debug_cout << "Found " << numfound << " forward tracks for this event!" << std::endl;
  
  return outputTracks;
}

//=============================================================================
void PrForward::find_forward_tracks(
  const VeloUTTracking::TrackVeloUT& veloUTTrack,
  std::vector<ForwardTracking::TrackForward>& outputTracks
) const {

  // Cache state information from state at the end of the VELO for
  // all subsequent processing
  FullState state_at_endvelo = veloUTTrack.state_endvelo;

  // The LHCb framework code had a PT preselection for the VeloUT tracks
  // here, which I am removing because this should be done explicitly through
  // track selectors if we do it at all, not hacked inside the tracking code

  // Some values related to the forward track which were stored in a dedicated
  // forward track class, let's see if I can get rid of that here
  const float zRef_track    = Forward::zReference;
  const float xAtRef = xFromVelo( zRef_track, state_at_endvelo );
  const float xParams_seed[4] = {xAtRef, state_at_endvelo.tx, 0.f, 0.f};
  const float yAtRef = yFromVelo( zRef_track, state_at_endvelo );
  const float yParams_seed[4] = {yAtRef, state_at_endvelo.ty, 0.f, 0.f};

  // First loop Hough cluster search, set initial search windows
  PrParameters pars_first{Forward::minXHits, Forward::maxXWindow, Forward::maxXWindowSlope, Forward::maxXGap, 4u};
  PrParameters pars_second{Forward::minXHits_2nd, Forward::maxXWindow_2nd, Forward::maxXWindowSlope_2nd, Forward::maxXGap_2nd, 4u};

  std::vector<int> allXHits[2];

  if(yAtRef>-5.f)collectAllXHits(&m_hits_layers, allXHits[1], xParams_seed, yParams_seed, state_at_endvelo, 1); 
  if(yAtRef< 5.f)collectAllXHits(&m_hits_layers, allXHits[0], xParams_seed, yParams_seed, state_at_endvelo, -1);

  std::vector<ForwardTracking::TrackForward> outputTracks1;
  
  if(yAtRef>-5.f)selectXCandidates(&m_hits_layers, allXHits[1], veloUTTrack, outputTracks1, zRef_track, 
				   xParams_seed, yParams_seed, state_at_endvelo, pars_first,  1);
  if(yAtRef< 5.f)selectXCandidates(&m_hits_layers, allXHits[0], veloUTTrack, outputTracks1, zRef_track, 
				   xParams_seed, yParams_seed, state_at_endvelo, pars_first, -1); 

  //debug_cout << "Found " << outputTracks1.size() << " X candidates in first loop" << std::endl;

  selectFullCandidates(outputTracks1,xParams_seed,yParams_seed, state_at_endvelo, pars_first);

  //debug_cout << "Found " << outputTracks1.size() << " full candidates in first loop" << std::endl;

  bool ok = std::any_of(outputTracks1.begin(), outputTracks1.end(),
                        [](const auto& track) {
                           return track.hitsNum > 10;
                        });

  std::vector<ForwardTracking::TrackForward> outputTracks2; 
  if (!ok && Forward::secondLoop) { // If you found nothing begin the 2nd loop
    if(yAtRef>-5.f)selectXCandidates(&m_hits_layers, allXHits[1], veloUTTrack, outputTracks2, zRef_track, 
				     xParams_seed, yParams_seed, state_at_endvelo, pars_second, 1);
    if(yAtRef< 5.f)selectXCandidates(&m_hits_layers, allXHits[0], veloUTTrack, outputTracks2, zRef_track, 
				     xParams_seed, yParams_seed, state_at_endvelo, pars_second, -1);  

    //debug_cout << "Found " << outputTracks1.size() << " X candidates in second loop" << std::endl;

    selectFullCandidates(outputTracks2,xParams_seed,yParams_seed, state_at_endvelo, pars_second);

    //debug_cout << "Found " << outputTracks1.size() << " full candidates in second loop" << std::endl;
    // Merge
    outputTracks1.insert(std::end(outputTracks1),
		 	 std::begin(outputTracks2),
	 		 std::end(outputTracks2));
    ok = not outputTracks1.empty();
  }
 
  //debug_cout << "About to do final arbitration of tracks " << ok << std::endl; 
  if(ok || !Forward::secondLoop){
    std::sort(outputTracks1.begin(), outputTracks1.end(), lowerByQuality );
    float minQuality = Forward::maxQuality;
    for ( auto& track : outputTracks1 ){
      //debug_cout << track.quality << " " << Forward::deltaQuality << " " << minQuality << std::endl;
      if(track.quality + Forward::deltaQuality < minQuality) minQuality = track.quality + Forward::deltaQuality;
      if(!(track.quality > minQuality)) {
        // add LHCbIDs from Velo and UT part of the track
        for ( int i_hit = 0; i_hit < veloUTTrack.track.hitsNum; ++i_hit ) {
          track.addLHCbID( veloUTTrack.track.LHCbIDs[i_hit] );
        }
        outputTracks.emplace_back(track);
        //debug_cout << "Found a forward track corresponding to a velo track!" << std::endl;
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
void PrForward::selectFullCandidates(std::vector<ForwardTracking::TrackForward>& outputTracks,
                                     const float xParams_seed[4],
                                     const float yParams_seed[4],
				     FullState state_at_endvelo,
				     PrParameters& pars ) const {

  std::vector<unsigned int> pc;
  std::vector<float> mlpInput(7, 0.); 

  std::vector<ForwardTracking::TrackForward> selectedTracks;

  for (std::vector<ForwardTracking::TrackForward>::iterator cand = std::begin(outputTracks);
       cand != std::end(outputTracks); ++cand) {
    // DvB: this bool is not used anywhere...??
    bool isValid = false; // In c++ this is in track class, try to understand why later
    pars.minStereoHits = 4;

    if(cand->hitsNum + pars.minStereoHits < Forward::minTotalHits) {
      pars.minStereoHits = Forward::minTotalHits - cand->hitsNum;
    }
    // search for hits in U/V layers
    std::vector<int> stereoHits = collectStereoHits(&m_hits_layers, *cand, state_at_endvelo, pars);
    debug_cout << "Collected " << stereoHits.size() << " valid stereo hits for full track search, with requirement of " << pars.minStereoHits << std::endl;
    if(stereoHits.size() < pars.minStereoHits) continue;
    // DIRTY HACK
    std::vector<std::pair<float,int> > tempforsort;
    tempforsort.clear();
    for (auto hit : stereoHits) { tempforsort.emplace_back(std::pair<float,int>(m_hits_layers.m_coord[hit],hit));}
    std::sort( tempforsort.begin(), tempforsort.end());
    stereoHits.clear();
    for (auto pair : tempforsort) {stereoHits.emplace_back(pair.second);}

    // select best U/V hits
    if ( !selectStereoHits(&m_hits_layers, *cand, stereoHits, state_at_endvelo, pars) ) continue;
    debug_cout << "Passed the stereo hits selection!" << std::endl;

    pc.clear();
    int planelist[12] = {0};
    // Hijacks LHCbIDs to store the values of the hits in the SoA for now, to be changed
    for (auto hit : cand->hit_indices) {
      pc.push_back(hit);
      planelist[m_hits_layers.m_planeCode[hit]/2] += 1;
    }
    
    //make a fit of ALL hits
    if(!fitXProjection(&m_hits_layers, cand->trackParams, pc, planelist, pars))continue;
    //debug_cout << "Passed the X projection fit" << std::endl;   
 
    //check in empty x layers for hits 
    auto checked_empty = (cand->trackParams[4]  < 0.f) ?
      addHitsOnEmptyXLayers(&m_hits_layers, cand->trackParams, xParams_seed, yParams_seed,
                              true, pc, planelist, pars, -1)
        : 
      addHitsOnEmptyXLayers(&m_hits_layers, cand->trackParams, xParams_seed, yParams_seed,
                              true, pc, planelist, pars, 1);

    if (not checked_empty) continue;
    //debug_cout << "Passed the empty check" << std::endl;

    //track has enough hits, calcualte quality and save if good enough
    //debug_cout << "Full track candidate has " << pc.size() << " hits on " << nbDifferent(planelist) << " different layers" << std::endl;    
    if(nbDifferent(planelist) >= Forward::minTotalHits){
      //debug_cout << "Computing final quality with NNs" << std::endl;

      const float qOverP  = calcqOverP(cand->trackParams[1], state_at_endvelo);
      //orig params before fitting , TODO faster if only calc once?? mem usage?
      const float xAtRef = cand->trackParams[0];
      float dSlope  = ( state_at_endvelo.x + (Forward::zReference - state_at_endvelo.z) * state_at_endvelo.tx - xAtRef ) / ( Forward::zReference - Forward::zMagnetParams[0]);
      const float zMagSlope = Forward::zMagnetParams[2] * pow(state_at_endvelo.tx,2) +  Forward::zMagnetParams[3] * pow(state_at_endvelo.ty,2);
      const float zMag    = Forward::zMagnetParams[0] + Forward::zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
      const float xMag    = state_at_endvelo.x + (zMag- state_at_endvelo.z) * state_at_endvelo.tx;
      const float slopeT  = ( xAtRef - xMag ) / ( Forward::zReference - zMag );
      dSlope        = slopeT - state_at_endvelo.tx;
      const float dyCoef  = dSlope * dSlope * state_at_endvelo.ty;

      float bx = slopeT;
      float ay = state_at_endvelo.y + (Forward::zReference - state_at_endvelo.z) * state_at_endvelo.ty;
      float by = state_at_endvelo.ty + dyCoef * Forward::byParams;

      //ay,by,bx params
      const float ay1  = cand->trackParams[4];
      const float by1  = cand->trackParams[5];
      const float bx1  = cand->trackParams[1];

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

      //debug_cout << "Track candidate has NN quality " << quality << std::endl;

      if(quality < Forward::maxQuality){
	cand->quality = quality;
	cand->hitsNum = pc.size();
        cand->hit_indices = pc;
        cand->set_qop( qOverP );
	// Must be a neater way to do this...
	selectedTracks.emplace_back(*cand);
      }
    }
  }
  outputTracks = selectedTracks;
}
