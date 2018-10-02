#include "PrForward.cuh"
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

__global__ void PrForward(
  const uint* dev_scifi_hits,
  const uint* dev_scifi_hit_count,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_states,
  VeloUTTracking::TrackUT * dev_veloUT_tracks,
  const int * dev_atomics_veloUT,
  SciFi::Track* dev_scifi_tracks,
  uint* dev_n_scifi_tracks ,
  SciFi::Tracking::TMVA* dev_tmva1,
  SciFi::Tracking::TMVA* dev_tmva2,
  SciFi::Tracking::Arrays* dev_constArrays  
) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks {(uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states {dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  // UT un-consolidated tracks (-> should be consolidated soon)
  const int* n_veloUT_tracks_event = dev_atomics_veloUT + event_number;
  VeloUTTracking::TrackUT* veloUT_tracks_event = dev_veloUT_tracks + event_number * VeloUTTracking::max_num_tracks;

  // SciFi un-consolidated types
  SciFi::Track* scifi_tracks_event = dev_scifi_tracks + event_number * SciFi::max_tracks;
  uint* n_scifi_tracks_event = dev_n_scifi_tracks + event_number;

  SciFi::SciFiHitCount scifi_hit_count;
  scifi_hit_count.typecast_after_prefix_sum((uint*) dev_scifi_hit_count, event_number, number_of_events);
  
  SciFi::SciFiHits scifi_hits;
  scifi_hits.typecast_sorted((uint*) dev_scifi_hits, scifi_hit_count.layer_offsets[number_of_events * SciFi::number_of_zones]);

  // initialize atomic SciFi tracks counter
  if ( threadIdx.x == 0 ) {
    *n_scifi_tracks_event = 0;
  }
  __syncthreads();
  
  // Loop over the veloUT input tracks
  for ( int i = 0; i < (*n_veloUT_tracks_event + blockDim.x - 1) / blockDim.x; ++i) {
    const int i_veloUT_track = i * blockDim.x + threadIdx.x;
    if ( i_veloUT_track < *n_veloUT_tracks_event ) {
      const VeloUTTracking::TrackUT& veloUTTr = veloUT_tracks_event[i_veloUT_track];
      
      const uint velo_states_index = event_tracks_offset + veloUTTr.veloTrackIndex;
      const MiniState velo_state {velo_states, velo_states_index};
      
      find_forward_tracks(
        scifi_hits,
        scifi_hit_count,
        veloUTTr,
        scifi_tracks_event,
        n_scifi_tracks_event,
        dev_tmva1,
        dev_tmva2,
        dev_constArrays,
        velo_state);
    }
  }
  
}

__host__ __device__ void find_forward_tracks(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  const VeloUTTracking::TrackUT& veloUTTrack,
  SciFi::Track* outputTracks,
  uint* n_forward_tracks,
  SciFi::Tracking::TMVA* tmva1,
  SciFi::Tracking::TMVA* tmva2,
  SciFi::Tracking::Arrays* constArrays,
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
  float coordX[2][SciFi::Tracking::max_x_hits];
  
  if(yAtRef>-5.f)
    collectAllXHits(
      scifi_hits, scifi_hit_count, allXHits[1], n_x_hits[1],
      coordX[1], xParams_seed, yParams_seed, constArrays,
      velo_state, veloUTTrack.qop, 1); 
  if(yAtRef< 5.f)
    collectAllXHits(
      scifi_hits, scifi_hit_count, allXHits[0], n_x_hits[0],
      coordX[0], xParams_seed, yParams_seed, constArrays,
      velo_state, veloUTTrack.qop, -1);


  SciFi::Tracking::Track candidate_tracks[SciFi::max_tracks];
  int n_candidate_tracks = 0;
  bool usedHits[SciFi::Constants::max_numhits_per_event] = { false };
  
  if(yAtRef>-5.f)selectXCandidates(
    scifi_hits, scifi_hit_count, allXHits[1], n_x_hits[1],
    usedHits, coordX[1], veloUTTrack,
    candidate_tracks, n_candidate_tracks,
    zRef_track, xParams_seed, yParams_seed,
    velo_state, pars_first,  constArrays, 1);
  if(yAtRef< 5.f)selectXCandidates(
    scifi_hits, scifi_hit_count, allXHits[0], n_x_hits[0],
    usedHits, coordX[0], veloUTTrack,
    candidate_tracks, n_candidate_tracks,
    zRef_track, xParams_seed, yParams_seed,
    velo_state, pars_first, constArrays, -1); 
 
  SciFi::Tracking::Track selected_tracks[SciFi::max_tracks];
  int n_selected_tracks = 0;
    
  selectFullCandidates(
    scifi_hits, scifi_hit_count,
    candidate_tracks,
    n_candidate_tracks,
    selected_tracks,
    n_selected_tracks,
    xParams_seed, yParams_seed,
    velo_state, veloUTTrack.qop,
    pars_first, tmva1, tmva2, constArrays, false);


  
  bool ok = false;
  for ( int i_track = 0; i_track < n_selected_tracks; ++i_track ) {
    if ( selected_tracks[i_track].hitsNum > 10 )
      ok = true;
  }

  SciFi::Tracking::Track candidate_tracks2[SciFi::Tracking::max_tracks_second_loop];
  int n_candidate_tracks2 = 0;
  
  if (!ok && SciFi::Tracking::secondLoop) { // If you found nothing begin the 2nd loop
    if(yAtRef>-5.f)selectXCandidates(
      scifi_hits, scifi_hit_count, allXHits[1], n_x_hits[1],
      usedHits, coordX[1], veloUTTrack,
      candidate_tracks2, n_candidate_tracks2,
      zRef_track, xParams_seed, yParams_seed,
      velo_state, pars_second, constArrays, 1);
    if(yAtRef< 5.f)selectXCandidates(
      scifi_hits, scifi_hit_count, allXHits[0], n_x_hits[0],
      usedHits, coordX[0], veloUTTrack,
      candidate_tracks2, n_candidate_tracks2,
      zRef_track, xParams_seed, yParams_seed,
      velo_state, pars_second, constArrays, -1);  

  
    
    SciFi::Tracking::Track selected_tracks2[SciFi::Tracking::max_tracks_second_loop];
    int n_selected_tracks2 = 0;
    
    selectFullCandidates(
      scifi_hits, scifi_hit_count,
      candidate_tracks2,
      n_candidate_tracks2,
      selected_tracks2,
      n_selected_tracks2,
      xParams_seed, yParams_seed,
      velo_state, veloUTTrack.qop,
      pars_second, tmva1, tmva2, constArrays, true);

    for ( int i_track = 0; i_track < n_selected_tracks2; ++i_track ) {
      assert( n_selected_tracks < SciFi::max_tracks);
      selected_tracks[n_selected_tracks++] = selected_tracks2[i_track];
    }

   
    
    ok = (n_selected_tracks > 0);
  }

  if(ok || !SciFi::Tracking::secondLoop){
    thrust::sort( thrust::seq, selected_tracks, selected_tracks + n_selected_tracks, lowerByQuality);
    float minQuality = SciFi::Tracking::maxQuality;
    for ( int i_track = 0; i_track < n_selected_tracks; ++i_track ) {
      SciFi::Tracking::Track& track = selected_tracks[i_track];
      if(track.quality + SciFi::Tracking::deltaQuality < minQuality)
        minQuality = track.quality + SciFi::Tracking::deltaQuality;
      if(!(track.quality > minQuality)) {
        
        SciFi::Track tr = makeTrack( track );
        // add LHCbIDs from Velo and UT part of the track
        for ( int i_hit = 0; i_hit < veloUTTrack.hitsNum; ++i_hit ) {
          tr.addLHCbID( veloUTTrack.LHCbIDs[i_hit] );
        }
        // add LHCbIDs from SciFi part of the track
        for ( int i_hit = 0; i_hit < track.hitsNum; ++i_hit ) {
          tr.addLHCbID( scifi_hits.LHCbID[ track.hit_indices[i_hit] ] );
        }
        
        assert(*n_forward_tracks < SciFi::max_tracks - 1);
#ifndef __CUDA_ARCH__
        outputTracks[(*n_forward_tracks)++] = tr;
#else
        uint n_tracks = atomicAdd(n_forward_tracks, 1);
        assert( n_tracks < SciFi::max_tracks );
        outputTracks[n_tracks] = tr;
#endif
      }
    }
  }
}

// Turn SciFi::Tracking::Track into a SciFi::Track
__host__ __device__ SciFi::Track makeTrack( SciFi::Tracking::Track track ) {
  SciFi::Track tr;
  tr.qop     = track.qop;
  tr.chi2    = track.chi2;

  return tr;
}

//=========================================================================
//  Create Full candidates out of xCandidates
//  Searching for stereo hits
//  Fit of all hits
//  save everything in track candidate folder
//=========================================================================
__host__ __device__ void selectFullCandidates(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  SciFi::Tracking::Track* candidate_tracks,
  int& n_candidate_tracks,
  SciFi::Tracking::Track* selected_tracks,
  int& n_selected_tracks,
  const float xParams_seed[4],
  const float yParams_seed[4],
  MiniState velo_state,
  const float VeloUT_qOverP,
  SciFi::Tracking::HitSearchCuts& pars,
  SciFi::Tracking::TMVA* tmva1,
  SciFi::Tracking::TMVA* tmva2,
  SciFi::Tracking::Arrays* constArrays,
  bool secondLoop)
{

  PlaneCounter planeCounter;
  planeCounter.clear();
  float mlpInput[7] = {0};
  
  for ( int i_track = 0; i_track < n_candidate_tracks; ++i_track ) {
    SciFi::Tracking::Track* cand = candidate_tracks + i_track;
    
    pars.minStereoHits = 4;

    if(cand->hitsNum + pars.minStereoHits < SciFi::Tracking::minTotalHits) {
      pars.minStereoHits = SciFi::Tracking::minTotalHits - cand->hitsNum;
    }
    
    // search for hits in U/V layers
    int stereoHits[SciFi::Tracking::max_stereo_hits];
    int n_stereoHits = 0;
    float stereoCoords[SciFi::Tracking::max_stereo_hits];
    collectStereoHits(
      scifi_hits, scifi_hit_count,
      *cand, velo_state,
      pars, constArrays, stereoCoords, 
      stereoHits, n_stereoHits);

    if(n_stereoHits < pars.minStereoHits) continue;
   
    // select best U/V hits
    if ( !selectStereoHits(
      scifi_hits, scifi_hit_count,
      *cand, constArrays,
      stereoCoords, stereoHits, n_stereoHits,
      velo_state, pars) ) continue;

    planeCounter.clear();
    for ( int i_hit = 0; i_hit < cand->hitsNum; ++i_hit ) {
      int hit = cand->hit_indices[i_hit];
      planeCounter.addHit( scifi_hits.planeCode[hit]/2 );
    }
    
    //make a fit of ALL hits
    if(!fitXProjection(scifi_hits, cand->trackParams, cand->hit_indices, cand->hitsNum, planeCounter, pars))continue;
 
    //track has enough hits, calcualte quality and save if good enough
    if(planeCounter.nbDifferent >= SciFi::Tracking::minTotalHits){

      const float qOverP  = calcqOverP(cand->trackParams[1], constArrays, velo_state);
      //orig params before fitting , TODO faster if only calc once?? mem usage?
      const float xAtRef = cand->trackParams[0];
      float dSlope  = ( velo_state.x + (SciFi::Tracking::zReference - velo_state.z) * velo_state.tx - xAtRef ) / ( SciFi::Tracking::zReference - constArrays->zMagnetParams[0]);
      const float zMagSlope = constArrays->zMagnetParams[2] * pow(velo_state.tx,2) +  constArrays->zMagnetParams[3] * pow(velo_state.ty,2);
      const float zMag    = constArrays->zMagnetParams[0] + constArrays->zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
      const float xMag    = velo_state.x + (zMag- velo_state.z) * velo_state.tx;
      const float slopeT  = ( xAtRef - xMag ) / ( SciFi::Tracking::zReference - zMag );
      dSlope        = slopeT - velo_state.tx;
      const float dyCoef  = dSlope * dSlope * velo_state.ty;

      float bx = slopeT;
      float ay = velo_state.y + (SciFi::Tracking::zReference - velo_state.z) * velo_state.ty;
      float by = velo_state.ty + dyCoef * SciFi::Tracking::byParams;

      //ay,by,bx params
      const float ay1  = cand->trackParams[4];
      const float by1  = cand->trackParams[5];
      const float bx1  = cand->trackParams[1];

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
      if(pars.minXHits > 4) quality = GetMvaValue(mlpInput, tmva1); //1st loop NN
      else                  quality = GetMvaValue(mlpInput, tmva2); //2nd loop NN

      quality = 1.f-quality; //backward compability

      if(quality < SciFi::Tracking::maxQuality){
        cand->quality = quality;
        cand->set_qop( qOverP );
	// Must be a neater way to do this...
        if (!secondLoop) assert (n_selected_tracks < SciFi::max_tracks );
        else if (secondLoop)assert (n_selected_tracks < SciFi::Tracking::max_tracks_second_loop );
        selected_tracks[n_selected_tracks++] = *cand;
      }
    }
  }
}
