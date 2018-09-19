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


//=============================================================================
// Main execution
//=============================================================================
void PrForward(
  SciFi::HitsSoA *hits_layers,
  const Velo::Consolidated::States& velo_states,
  const uint event_tracks_offset,
  const VeloUTTracking::TrackUT * veloUT_tracks,
  const int n_veloUT_tracks,
  const SciFi::TMVA1& tmva1,
  const SciFi::TMVA2& tmva2,
  SciFi::Track outputTracks[SciFi::max_tracks],
  int& n_forward_tracks)
{
  
  int numfound = 0;
  // Loop over the veloUT input tracks
  for ( int i_veloUT_track = 0; i_veloUT_track < n_veloUT_tracks; ++i_veloUT_track ) {
    const VeloUTTracking::TrackUT& veloUTTr = veloUT_tracks[i_veloUT_track];

    const uint velo_states_index = event_tracks_offset + veloUTTr.veloTrackIndex;
    const MiniState velo_state {velo_states, velo_states_index};
    
    find_forward_tracks(
      hits_layers,
      veloUTTr,
      outputTracks,
      n_forward_tracks,
      tmva1,
      tmva2,
      velo_state);
    
    
    // Reset used hits etc.
    // these should not be part of the HitsSoA struct
    // not thread safe!!
    for (int i =0; i< SciFi::Constants::max_numhits_per_event; i++){
      hits_layers->m_used[i] = false;
      hits_layers->m_coord[i] = 0.0;
    }
  }

}

//=============================================================================
void find_forward_tracks(
  SciFi::HitsSoA* hits_layers,
  const VeloUTTracking::TrackUT& veloUTTrack,
  SciFi::Track outputTracks[SciFi::max_tracks],
  int& n_forward_tracks,
  const SciFi::TMVA1& tmva1,
  const SciFi::TMVA2& tmva2,
  const MiniState& velo_state
) {

  // The LHCb framework code had a PT preselection for the VeloUT tracks
  // here, which I am removing because this should be done explicitly through
  // track selectors if we do it at all, not hacked inside the tracking code

  // Some values related to the forward track which were stored in a dedicated
  // forward track class, let's see if I can get rid of that here
  const float zRef_track    = SciFi::Tracking::zReference;
  const float xAtRef = xFromVelo( zRef_track, velo_state );
  const float xParams_seed[4] = {xAtRef, velo_state.tx, 0.f, 0.f};
  const float yAtRef = yFromVelo( zRef_track, velo_state );
  const float yParams_seed[4] = {yAtRef, velo_state.ty, 0.f, 0.f};

  // First loop Hough cluster search, set initial search windows
  SciFi::Tracking::HitSearchCuts pars_first{SciFi::Tracking::minXHits, SciFi::Tracking::maxXWindow, SciFi::Tracking::maxXWindowSlope, SciFi::Tracking::maxXGap, 4u};
  SciFi::Tracking::HitSearchCuts pars_second{SciFi::Tracking::minXHits_2nd, SciFi::Tracking::maxXWindow_2nd, SciFi::Tracking::maxXWindowSlope_2nd, SciFi::Tracking::maxXGap_2nd, 4u};

  int allXHits[2][SciFi::Tracking::max_x_hits];
  int n_x_hits[2] = {};
  
  if(yAtRef>-5.f)collectAllXHits(hits_layers, allXHits[1], n_x_hits[1], xParams_seed, yParams_seed, velo_state, veloUTTrack.qop, 1); 
  if(yAtRef< 5.f)collectAllXHits(hits_layers, allXHits[0], n_x_hits[0], xParams_seed, yParams_seed, velo_state, veloUTTrack.qop, -1);

  SciFi::Track candidate_tracks[SciFi::max_tracks];
  int n_candidate_tracks = 0;
  
  if(yAtRef>-5.f)selectXCandidates(hits_layers, allXHits[1], n_x_hits[1], veloUTTrack, candidate_tracks, n_candidate_tracks, zRef_track, 
				   xParams_seed, yParams_seed, velo_state, pars_first,  1);
  if(yAtRef< 5.f)selectXCandidates(hits_layers, allXHits[0], n_x_hits[0], veloUTTrack, candidate_tracks, n_candidate_tracks, zRef_track, 
				   xParams_seed, yParams_seed, velo_state, pars_first, -1); 

  SciFi::Track selected_tracks[SciFi::max_tracks];
  int n_selected_tracks = 0;
    
  selectFullCandidates(
    hits_layers,
    candidate_tracks,
    n_candidate_tracks,
    selected_tracks,
    n_selected_tracks,
    xParams_seed, yParams_seed,
    velo_state, veloUTTrack.qop,
    pars_first, tmva1, tmva2, false);

   bool ok = false;
  for ( int i_track = 0; i_track < n_selected_tracks; ++i_track ) {
    if ( selected_tracks[i_track].hitsNum > 10 )
      ok = true;
  }

  SciFi::Track candidate_tracks2[SciFi::Tracking::max_tracks_second_loop];
  int n_candidate_tracks2 = 0;
  
  if (!ok && SciFi::Tracking::secondLoop) { // If you found nothing begin the 2nd loop
    if(yAtRef>-5.f)selectXCandidates(hits_layers, allXHits[1], n_x_hits[1], veloUTTrack, candidate_tracks2, n_candidate_tracks2, zRef_track, 
				     xParams_seed, yParams_seed, velo_state, pars_second, 1);
    if(yAtRef< 5.f)selectXCandidates(hits_layers, allXHits[0], n_x_hits[0], veloUTTrack, candidate_tracks2, n_candidate_tracks2, zRef_track, 
				     xParams_seed, yParams_seed, velo_state, pars_second, -1);  

    SciFi::Track selected_tracks2[SciFi::Tracking::max_tracks_second_loop];
    int n_selected_tracks2 = 0;
    
    selectFullCandidates(
      hits_layers,
      candidate_tracks2,
      n_candidate_tracks2,
      selected_tracks2,
      n_selected_tracks2,
      xParams_seed, yParams_seed,
      velo_state, veloUTTrack.qop,
      pars_second, tmva1, tmva2, true);

    for ( int i_track = 0; i_track < n_selected_tracks2; ++i_track ) {
      assert( n_selected_tracks < SciFi::max_tracks);
      selected_tracks[n_selected_tracks++] = selected_tracks2[i_track];
    }
    
    ok = (n_selected_tracks > 0);
  }
 
  //debug_cout << "About to do final arbitration of tracks " << ok << std::endl; 
  if(ok || !SciFi::Tracking::secondLoop){
    std::sort( selected_tracks, selected_tracks + n_selected_tracks, lowerByQuality);
    float minQuality = SciFi::Tracking::maxQuality;
    for ( int i_track = 0; i_track < n_selected_tracks; ++i_track ) {
      SciFi::Track& track = selected_tracks[i_track];
      //debug_cout << track.quality << " " << SciFi::Tracking::deltaQuality << " " << minQuality << std::endl;
      if(track.quality + SciFi::Tracking::deltaQuality < minQuality)
        minQuality = track.quality + SciFi::Tracking::deltaQuality;
      if(!(track.quality > minQuality)) {
        // add LHCbIDs from Velo and UT part of the track
        for ( int i_hit = 0; i_hit < veloUTTrack.hitsNum; ++i_hit ) {
          track.addLHCbID( veloUTTrack.LHCbIDs[i_hit] );
        }
        assert(n_forward_tracks < SciFi::max_tracks);
        outputTracks[n_forward_tracks++] = track;
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
void selectFullCandidates(
  SciFi::HitsSoA* hits_layers,
  SciFi::Track* candidate_tracks,
  int& n_candidate_tracks,
  SciFi::Track* selected_tracks,
  int& n_selected_tracks,
  const float xParams_seed[4],
  const float yParams_seed[4],
  MiniState velo_state,
  const float VeloUT_qOverP,
  SciFi::Tracking::HitSearchCuts& pars,
  const SciFi::TMVA1& tmva1,
  const SciFi::TMVA2& tmva2,
  bool secondLoop)
{

  PlaneCounter planeCounter;
  float mlpInput[7] = {0};
  
  for ( int i_track = 0; i_track < n_candidate_tracks; ++i_track ) {
    SciFi::Track& cand = candidate_tracks[i_track];
    
    pars.minStereoHits = 4;

    if(cand.hitsNum + pars.minStereoHits < SciFi::Tracking::minTotalHits) {
      pars.minStereoHits = SciFi::Tracking::minTotalHits - cand.hitsNum;
    }
    
    // search for hits in U/V layers
    std::vector<unsigned int> stereoHits = collectStereoHits(hits_layers, cand, velo_state, pars);
    // debug_cout << "Collected " << stereoHits.size() << " valid stereo hits for full track search, with requirement of " << pars.minStereoHits << std::endl;
    if(stereoHits.size() < pars.minStereoHits) continue;
    // DIRTY HACK
    std::vector<std::pair<float,int> > tempforsort;
    tempforsort.clear();
    for (auto hit : stereoHits) { tempforsort.emplace_back(std::pair<float,int>(hits_layers->m_coord[hit],hit));}
    std::sort( tempforsort.begin(), tempforsort.end());
    stereoHits.clear();
    for (auto pair : tempforsort) {stereoHits.emplace_back(pair.second);}
    //debug_cout << "# of collected stereo hits = " << int(stereoHits.size()) << std::endl;
      
    // select best U/V hits
    if ( !selectStereoHits(hits_layers, cand, stereoHits, velo_state, pars) ) continue;
    //debug_cout << "Passed the stereo hits selection!" << std::endl;

    planeCounter.clear();
    for (auto hit : cand.hit_indices) {
      planeCounter.addHit( hits_layers->m_planeCode[hit]/2 );
    }
    
    //make a fit of ALL hits
    if(!fitXProjection(hits_layers, cand.trackParams, cand.hit_indices, planeCounter, pars))continue;
    //debug_cout << "Passed the X projection fit" << std::endl;   
 
    //check in empty x layers for hits 
    auto checked_empty = (cand.trackParams[4]  < 0.f) ?
      addHitsOnEmptyXLayers(hits_layers, cand.trackParams, xParams_seed, yParams_seed,
                              true, cand.hit_indices, planeCounter, pars, -1)
        : 
      addHitsOnEmptyXLayers(hits_layers, cand.trackParams, xParams_seed, yParams_seed,
                              true, cand.hit_indices, planeCounter, pars, 1);

    if (not checked_empty) continue;
    //debug_cout << "Passed the empty check" << std::endl;

    //track has enough hits, calcualte quality and save if good enough
    //debug_cout << "Full track candidate has " << pc.size() << " hits on " << nbDifferent(planelist) << " different layers" << std::endl;    
    if(planeCounter.nbDifferent >= SciFi::Tracking::minTotalHits){
      //debug_cout << "Computing final quality with NNs" << std::endl;

      const float qOverP  = calcqOverP(cand.trackParams[1], velo_state);
      //orig params before fitting , TODO faster if only calc once?? mem usage?
      const float xAtRef = cand.trackParams[0];
      float dSlope  = ( velo_state.x + (SciFi::Tracking::zReference - velo_state.z) * velo_state.tx - xAtRef ) / ( SciFi::Tracking::zReference - SciFi::Tracking::zMagnetParams[0]);
      const float zMagSlope = SciFi::Tracking::zMagnetParams[2] * pow(velo_state.tx,2) +  SciFi::Tracking::zMagnetParams[3] * pow(velo_state.ty,2);
      const float zMag    = SciFi::Tracking::zMagnetParams[0] + SciFi::Tracking::zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
      const float xMag    = velo_state.x + (zMag- velo_state.z) * velo_state.tx;
      const float slopeT  = ( xAtRef - xMag ) / ( SciFi::Tracking::zReference - zMag );
      dSlope        = slopeT - velo_state.tx;
      const float dyCoef  = dSlope * dSlope * velo_state.ty;

      float bx = slopeT;
      float ay = velo_state.y + (SciFi::Tracking::zReference - velo_state.z) * velo_state.ty;
      float by = velo_state.ty + dyCoef * SciFi::Tracking::byParams;

      //ay,by,bx params
      const float ay1  = cand.trackParams[4];
      const float by1  = cand.trackParams[5];
      const float bx1  = cand.trackParams[1];

      mlpInput[0] = planeCounter.nbDifferent;
      mlpInput[1] = qOverP;
      mlpInput[2] = VeloUT_qOverP - qOverP; //veloUT - scifi
      if(std::fabs(VeloUT_qOverP) < 1e-9f) mlpInput[2] = 0.f; //no momentum estiamte
      mlpInput[3] = pow(velo_state.tx,2) + pow(velo_state.ty,2);
      mlpInput[4] = by - by1;
      mlpInput[5] = bx - bx1;
      mlpInput[6] = ay - ay1;

      float quality = 0.f;
      /// WARNING: if the NN classes straight out of TMVA are used, put a mutex here!
      if(pars.minXHits > 4) quality = tmva1.GetMvaValue(mlpInput); //1st loop NN
      else                   quality = tmva2.GetMvaValue(mlpInput); //2nd loop NN

      quality = 1.f-quality; //backward compability

      //debug_cout << "Track candidate has " << pars.minXHits << " minXHits and NN quality " << quality << std::endl;

      if(quality < SciFi::Tracking::maxQuality){
        cand.quality = quality;
        cand.set_qop( qOverP );
	// Must be a neater way to do this...
        if (!secondLoop) assert (n_selected_tracks < SciFi::max_tracks );
        else if (secondLoop)assert (n_selected_tracks < SciFi::Tracking::max_tracks_second_loop );
        selected_tracks[n_selected_tracks++] = cand;
      }
    }
  }
  //outputTracks = selectedTracks;
}
