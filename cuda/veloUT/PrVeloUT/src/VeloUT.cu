#include "VeloUT.cuh"

__global__ void veloUT(
  VeloUTTracking::HitsSoA* dev_ut_hits,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states,
  VeloUTTracking::TrackUT* dev_veloUT_tracks,
  int* dev_n_veloUT_tracks,
  PrUTMagnetTool* dev_ut_magnet_tool
) {

  if ( threadIdx.x == 0 ) {
    
  const int number_of_events = gridDim.x;
  const int event_number = blockIdx.x;
  
  const int number_of_tracks_event = *(dev_atomics_storage + event_number);
  const int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  const int accumulated_tracks_event = accumulated_tracks_base_pointer[event_number];
  VeloUTTracking::HitsSoA* hits_layers_event = dev_ut_hits + event_number;
  VeloState* velo_states_event = dev_velo_states + accumulated_tracks_event;
  int* n_veloUT_tracks_event = dev_n_veloUT_tracks + event_number;
  VeloUTTracking::TrackUT* veloUT_tracks_event = dev_veloUT_tracks + event_number * VeloUTTracking::max_num_tracks;
  
  // initialize atomic veloUT tracks counter
  *n_veloUT_tracks_event = 0;
  __syncthreads();

  int posLayers[4][85];
  fillIterators(hits_layers_event, posLayers);

  const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  const float* bdlTable     = &(dev_ut_magnet_tool->bdlTable[0]);

  // array to store indices of selected hits in layers
  // -> can then access the hit information in the HitsSoA
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers];
  
  for ( int i_track = 0; i_track < number_of_tracks_event; ++i_track ) {
    if ( velo_states_event[i_track].backward ) continue;
    
    if( !veloTrackInUTAcceptance( velo_states_event[i_track] ) ) continue;
    //n_velo_tracks_in_UT++;
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      n_hitCandidatesInLayers[i_layer] = 0;
    }
    if( !getHits(
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          posLayers,
          hits_layers_event,
          fudgeFactors,
          velo_states_event[i_track] )
        ) continue;
    
    TrackHelper helper(velo_states_event[i_track]);
    
    // go through UT layers in forward direction
    if( !formClusters(
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          hits_layers_event,
          helper,
          true) ){
      
      // go through UT layers in backward direction
      formClusters(
        hitCandidatesInLayers,
        n_hitCandidatesInLayers,
        hits_layers_event,
        helper,
        false);
    }
    
    if ( helper.n_hits > 0 ) {
      prepareOutputTrack(
        dev_velo_track_hit_number,
        dev_velo_track_hits,
        accumulated_tracks_event,
        i_track,
        helper,
        hitCandidatesInLayers,
        n_hitCandidatesInLayers,
        hits_layers_event,
        veloUT_tracks_event,
        n_veloUT_tracks_event,
        bdlTable);
    }
    
  } // tracks
  
  //printf("event %u has %u tracks \n", event_number, *n_veloUT_tracks_event);
  // for ( int i_track = 0; i_track < *n_veloUT_tracks_event; ++i_track ) {
  //   printf("in event %u, at track %u, # of hits = %u \n", event_number, i_track, veloUT_tracks_event[i_track].hitsNum);
  // }
  
  }
  
}
