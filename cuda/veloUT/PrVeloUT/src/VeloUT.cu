#include "VeloUT.cuh"

__global__ void veloUT(
  VeloUTTracking::HitsSoA* dev_ut_hits,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states,
  VeloUTTracking::TrackUT* dev_veloUT_tracks,
  int* dev_atomics_veloUT,
  PrUTMagnetTool* dev_ut_magnet_tool
) {

  const int number_of_events = gridDim.x;
  const int event_number = blockIdx.x;
  
  const int number_of_tracks_event = *(dev_atomics_storage + event_number);
  const int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  const int accumulated_tracks_event = accumulated_tracks_base_pointer[event_number];
  VeloUTTracking::HitsSoA* hits_layers_event = dev_ut_hits + event_number;
  VeloState* velo_states_event = dev_velo_states + accumulated_tracks_event;
  /* dev_atomics_veloUT contains in an SoA:
     1. # of veloUT tracks
     2. # velo tracks in UT acceptance
  */
  int* n_veloUT_tracks_event = dev_atomics_veloUT + event_number;
  VeloUTTracking::TrackUT* veloUT_tracks_event = dev_veloUT_tracks + event_number * VeloUTTracking::max_num_tracks;
  int* n_velo_tracks_in_UT_event = dev_atomics_veloUT + number_of_events + event_number;
  
  // initialize atomic veloUT tracks counter
  *n_veloUT_tracks_event = 0;
  *n_velo_tracks_in_UT_event = 0;
  __syncthreads();

  int posLayers[4][85];
  fillIterators(hits_layers_event, posLayers);

  const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  const float* bdlTable     = &(dev_ut_magnet_tool->bdlTable[0]);

  // array to store indices of selected hits in layers
  // -> can then access the hit information in the HitsSoA
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers];

  for ( int i = 0; i < (number_of_tracks_event + blockDim.x - 1) / blockDim.x; ++i) {
    const int i_track = i * blockDim.x + threadIdx.x;
    if ( i_track >= number_of_tracks_event ) continue;
         
    if ( velo_states_event[i_track].backward ) continue;
    
    if( !veloTrackInUTAcceptance( velo_states_event[i_track] ) ) continue;
    //atomicAdd(n_velo_tracks_in_UT_event, 1);

    // for storing calculated x and z positions of hits for this track
    float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
    float z_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
    
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      n_hitCandidatesInLayers[i_layer] = 0;
    }
    if( !getHits(
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          x_pos_layers,
          z_pos_layers,
          posLayers,
          hits_layers_event,
          fudgeFactors,
          velo_states_event[i_track] )
        ) continue;

    __syncthreads(); // due to writing of x and z values
    //atomicAdd(n_velo_tracks_in_UT_event, 1);
    
    TrackHelper helper(velo_states_event[i_track]);

    int hitCandidateIndices[VeloUTTracking::n_layers];
    // go through UT layers in forward direction
    if( !formClusters(
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          x_pos_layers,
          z_pos_layers,
          hitCandidateIndices,
          hits_layers_event,
          helper,
          true, n_velo_tracks_in_UT_event) ){
      
      // go through UT layers in backward direction
      formClusters(
        hitCandidatesInLayers,
        n_hitCandidatesInLayers,
        x_pos_layers,
        z_pos_layers,
        hitCandidateIndices,
        hits_layers_event,
        helper,
        false, n_velo_tracks_in_UT_event);
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
        x_pos_layers,
        z_pos_layers,
        hitCandidateIndices,
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
