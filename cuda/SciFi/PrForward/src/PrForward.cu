// *********************************************************************************
// ************************ Introduction to Forward Tracking **********************
// *********************************************************************************
//
//  A detailed introduction in Forward tracking (with real pictures!) can be
//  found here:
//  (2002) http://cds.cern.ch/record/684710/files/lhcb-2002-008.pdf
//  (2007) http://cds.cern.ch/record/1033584/files/lhcb-2007-015.pdf
//  (2014) http://cds.cern.ch/record/1641927/files/LHCb-PUB-2014-001.pdf
//
// *** Short Introduction in geometry:
//
// The SciFi Tracker Detector, or simple Fibre Tracker (FT) consits out of 3 stations.
// Each station consists out of 4 planes/layers. Thus there are in total 12 layers,
// in which a particle can leave a hit. The reasonable maximum number of hits a track
// can have is thus also 12 (sometimes 2 hits per layer are picked up).
//
// Each layer consists out of several Fibre mats. A fibre has a diameter of below a mm.(FIXME)
// Several fibres are glued alongside each other to form a mat.
// A Scintilating Fibre produces light, if a particle traverses. This light is then
// detected on the outside of the Fibre mat.
//
// Looking from the collision point, one (X-)layer looks like the following:
//
//                    y       6m
//                    ^  ||||||||||||| Upper side
//                    |  ||||||||||||| 2.5m
//                    |  |||||||||||||
//                   -|--||||||o||||||----> -x
//                       |||||||||||||
//                       ||||||||||||| Lower side
//                       ||||||||||||| 2.5m
//
// All fibres are aranged parallel to the y-axis. There are three different
// kinds of layers, denoted by X,U,V. The U/V layers are rotated with respect to
// the X-layers by +/- 5 degrees, to also get a handle of the y position of the
// particle. As due to the magnetic field particles are only deflected in
// x-direction, this configuration offers the best resolution.
// The layer structure in the FT is XUVX-XUVX-XUVX.
//
// The detector is divided into an upeer and a lower side (>/< y=0). As particles
// are only deflected in x direction there are only very(!) few particles that go
// from the lower to the upper side, or vice versa. The reconstruction algorithm
// can therefore be split into two independent steps: First track reconstruction
// for tracks in the upper side, and afterwards for tracks in the lower side.
//
// Due to construction issues this is NOT true for U/V layers. In these layers the
// complete(!) fibre modules are rotated, producing a zic-zac pattern at y=0, also
// called  "the triangles". Therefore for U/V layers it must be explicetly also
// searched for these hit on the "other side", if the track is close to y=0.
// Sketch (rotation exagerated!):
//                                          _.*
//     y ^   _.*                         _.*
//       | .*._      Upper side       _.*._
//       |     *._                 _.*     *._
//       |--------*._           _.*           *._----------------> x
//       |           *._     _.*                 *._     _.*
//                      *._.*       Lower side      *._.*
//
//
//
//
//
//       Zone ordering defined on PrKernel/PrFTInfo.h
//
//     y ^
//       |    1  3  5  7     9 11 13 15    17 19 21 23
//       |    |  |  |  |     |  |  |  |     |  |  |  |
//       |    x  u  v  x     x  u  v  x     x  u  v  x   <-- type of layer
//       |    |  |  |  |     |  |  |  |     |  |  |  |
//       |------------------------------------------------> z
//       |    |  |  |  |     |  |  |  |     |  |  |  |
//       |    |  |  |  |     |  |  |  |     |  |  |  |
//       |    0  2  4  6     8 10 12 14    16 18 20 22
//
//
// *** Short introduction in the Forward Tracking algorithm
//
// The track reconstruction is seperated into several steps:
//
// 1) Using only X-hits
//    1.1) Preselection: collectAllXHits()
//    1.2) Hough Transformation: xAtRef_SamePlaneHits()
//    1.3) Cluster search: selectXCandidates()
//    1.4) Linear and than Cubic Fit of X-Projection
// 2) Introducing U/V hits or also called stereo hits
//    2.1) Preselection: collectStereoHits
//    2.2) Cluster search: selectStereoHits
//    2.3) Fit Y-Projection
// 3) Using all (U+V+X) hits
//    3.1) Fitting X-Projection
//    3.2) calculating track quality with a Neural Net
//    3.3) final clone+ghost killing
//
// *****************************************************************

#include "PrForward.cuh"

//-----------------------------------------------------------------------------
// Implementation file for class : PrForward
//
// Based on code written by :
// 2012-03-20 : Olivier Callot
// 2013-03-15 : Thomas Nikodem
// 2015-02-13 : Sevda Esen [additional search in the triangles by Marian Stahl]
// 2016-03-09 : Thomas Nikodem [complete restructuring]
// 2018-08    : Vava Gligorov [extract code from Rec, make compile within GPU framework
// 2018-09    : Dorothea vom Bruch [convert to CUDA, runs on GPU]
//-----------------------------------------------------------------------------

//=============================================================================

// Kernel to call Forward tracking on GPU
// Loop over veloUT input tracks using threadIdx.x
__global__ void PrForward(
  const uint* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
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

  // SciFi un-consolidated track types
  SciFi::Track* scifi_tracks_event = dev_scifi_tracks + event_number * SciFi::max_tracks;
  uint* n_scifi_tracks_event = dev_n_scifi_tracks + event_number;

  // SciFi hits
  const uint total_number_of_hits = dev_scifi_hit_count[number_of_events * SciFi::Constants::n_zones];
  SciFi::SciFiHitCount scifi_hit_count;
  scifi_hit_count.typecast_after_prefix_sum((uint*) dev_scifi_hit_count, event_number, number_of_events);
  SciFi::SciFiHits scifi_hits;
  scifi_hits.typecast_sorted((uint*)dev_scifi_hits, total_number_of_hits);

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

/* Look first in x layers, then in stereo layers for hits
   do 1D Hough transform for x- and stereo hits
   do global 1D Hough transform
   use TMVAs to obtain track quality */
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
  int n_x_hits[2] = {0};
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

  SciFi::Tracking::Track candidate_tracks[SciFi::Tracking::max_candidate_tracks];
  int n_candidate_tracks = 0;
  bool usedHits[2][SciFi::Tracking::max_x_hits] = {false};
  
  if(yAtRef>-5.f)selectXCandidates(
    scifi_hits, scifi_hit_count, allXHits[1], n_x_hits[1],
    usedHits[1], coordX[1], veloUTTrack,
    candidate_tracks, n_candidate_tracks,
    zRef_track, xParams_seed, yParams_seed,
    velo_state, pars_first,  constArrays, 1, false);
  if(yAtRef< 5.f)selectXCandidates(
    scifi_hits, scifi_hit_count, allXHits[0], n_x_hits[0],
    usedHits[0], coordX[0], veloUTTrack,
    candidate_tracks, n_candidate_tracks,
    zRef_track, xParams_seed, yParams_seed,
    velo_state, pars_first, constArrays, -1, false); 
  
  SciFi::Tracking::Track selected_tracks[SciFi::Tracking::max_selected_tracks];
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
  assert( n_selected_tracks < SciFi::Tracking::max_selected_tracks );

  SciFi::Tracking::Track candidate_tracks2[SciFi::Tracking::max_tracks_second_loop];
  int n_candidate_tracks2 = 0;

  if (!ok && SciFi::Tracking::secondLoop) { // If you found nothing begin the 2nd loop
    if(yAtRef>-5.f)selectXCandidates(
      scifi_hits, scifi_hit_count, allXHits[1], n_x_hits[1],
      usedHits[1], coordX[1], veloUTTrack,
      candidate_tracks2, n_candidate_tracks2,
      zRef_track, xParams_seed, yParams_seed,
      velo_state, pars_second, constArrays, 1, true);
    if(yAtRef< 5.f)selectXCandidates(
      scifi_hits, scifi_hit_count, allXHits[0], n_x_hits[0],
      usedHits[0], coordX[0], veloUTTrack,
      candidate_tracks2, n_candidate_tracks2,
      zRef_track, xParams_seed, yParams_seed,
      velo_state, pars_second, constArrays, -1, true);  

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
      assert( n_selected_tracks < SciFi::Tracking::max_selected_tracks );
      selected_tracks[n_selected_tracks++] = selected_tracks2[i_track];
    }
   
    ok = (n_selected_tracks > 0);
  }
 
  if(ok || !SciFi::Tracking::secondLoop){

    if ( n_selected_tracks > 1 ) {
      // not using thrust::sort due to temporary_buffer::allocate:: get_temporary_buffer failed" error
      //thrust::sort( thrust::seq, selected_tracks, selected_tracks + n_selected_tracks, lowerByQuality);
      sort_tracks( 
        selected_tracks, 
        n_selected_tracks,
        [] (SciFi::Tracking::Track t1, SciFi::Tracking::Track t2) {
          if ( t1.quality < t2.quality ) return -1;
          if ( t1.quality == t2.quality ) return 0;
          return 1;
        }
      );
      
    }

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
        if ( tr.hitsNum >= VeloUTTracking::max_track_size )
          printf("veloUT track hits Num = %u \n", tr.hitsNum );
        assert( tr.hitsNum < VeloUTTracking::max_track_size );
        
        // add LHCbIDs from SciFi part of the track
        for ( int i_hit = 0; i_hit < track.hitsNum; ++i_hit ) {
          tr.addLHCbID( scifi_hits.LHCbID[ track.hit_indices[i_hit] ] );
        }
        assert( tr.hitsNum < SciFi::max_track_size );

        if ( *n_forward_tracks >= SciFi::max_tracks )
          printf("n_forward_tracks = %u \n", *n_forward_tracks);
        assert(*n_forward_tracks < SciFi::max_tracks );
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
    
    int stereoHits[SciFi::Tracking::max_stereo_hits];
    int n_stereoHits = 0;
    float stereoCoords[SciFi::Tracking::max_stereo_hits];
    collectStereoHits(
      scifi_hits, scifi_hit_count,
      *cand, velo_state,
      pars, constArrays, stereoCoords, 
      stereoHits, n_stereoHits);

    if(n_stereoHits < pars.minStereoHits) continue;
    
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
    
    //make a fit of ALL hits using their x coordinate
    if(!quadraticFitX(scifi_hits, cand->trackParams, cand->hit_indices, cand->hitsNum, planeCounter, pars))continue;
 
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
      if(fabsf(VeloUT_qOverP) < 1e-9f) mlpInput[2] = 0.f; //no momentum estiamte
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
        if (!secondLoop) 
          assert (n_selected_tracks < SciFi::Tracking::max_selected_tracks );
        else if (secondLoop)
          assert (n_selected_tracks < SciFi::Tracking::max_tracks_second_loop );
        selected_tracks[n_selected_tracks++] = *cand;
        if (!secondLoop) {
          if ( n_selected_tracks >= SciFi::Tracking::max_selected_tracks ) break;
        }
        else if ( secondLoop ) {
          if ( n_selected_tracks >= SciFi::Tracking::max_tracks_second_loop ) break;
        }
          
      }  
    }
  }
}
