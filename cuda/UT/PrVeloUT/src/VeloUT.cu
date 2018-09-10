#include "VeloUT.cuh"

__global__ void veloUT(
  uint* dev_ut_hits,
  uint* dev_ut_hit_count,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states,
  VeloUTTracking::TrackUT* dev_veloUT_tracks,
  int* dev_atomics_veloUT,
  PrUTMagnetTool* dev_ut_magnet_tool,
  float* dev_ut_dxDy
) {
  const int number_of_events = gridDim.x;
  const int event_number = blockIdx.x;
  
  const int number_of_tracks_event = *(dev_atomics_storage + event_number);
  const int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  const int accumulated_tracks_event = accumulated_tracks_base_pointer[event_number];
  VeloState* velo_states_event = dev_velo_states + accumulated_tracks_event;

  UTHits ut_hits;
  ut_hits.typecast_sorted(dev_ut_hits, dev_ut_hit_count[number_of_events * VeloUTTracking::n_layers]);
  // VeloUTTracking::HitsSoA* hits_layers_event = dev_ut_hits + event_number;

  uint* layer_offsets = dev_ut_hit_count + event_number * VeloUTTracking::n_layers;
  uint* n_hits_layers = dev_ut_hit_count + number_of_events * VeloUTTracking::n_layers + 1 + event_number * VeloUTTracking::n_layers;

  /* dev_atomics_veloUT contains in an SoA:
     1. # of veloUT tracks
     2. # velo tracks in UT acceptance
  */
  int* n_veloUT_tracks_event = dev_atomics_veloUT + event_number;
  VeloUTTracking::TrackUT* veloUT_tracks_event = dev_veloUT_tracks + event_number * VeloUTTracking::max_num_tracks;
  int* n_velo_tracks_in_UT_event = dev_atomics_veloUT + number_of_events + event_number;
  
  // initialize atomic veloUT tracks counter
  if ( threadIdx.x == 0 ) {
    *n_veloUT_tracks_event = 0;
    *n_velo_tracks_in_UT_event = 0;
  }
  __syncthreads();

  int posLayers[4][85];

  // printf("first hit: cos = %f, yBegin = %f, yEnd = %f, zAtYEq0 = %f, xAtYEq0 = %f, weight = %f, highThreshold = %u \n",
  //        hits_layers_event->cos(0),
  //        hits_layers_event->yBegin(0),
  //        hits_layers_event->yEnd(0),
  //        hits_layers_event->zAtYEq0(0),
  //        hits_layers_event->xAtYEq0(0),
  //        hits_layers_event->weight(0),
  //        hits_layers_event->highThreshold(0));
         
  fillIterators(ut_hits, layer_offsets, n_hits_layers, posLayers);

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
    atomicAdd(n_velo_tracks_in_UT_event, 1);

     // for storing calculated x position of hits for this track
    float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];

    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      n_hitCandidatesInLayers[i_layer] = 0;
    }
    if( !getHits(
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          x_pos_layers,
          posLayers,
          ut_hits,
          layer_offsets,
          n_hits_layers,
          fudgeFactors,
          velo_states_event[i_track],
          dev_ut_dxDy )
        ) continue;

    TrackHelper helper(velo_states_event[i_track]);

    // indices within hitCandidatesInLayers for selected hits belonging to best track 
    int hitCandidateIndices[VeloUTTracking::n_layers];
    
    // go through UT layers in forward direction
    if( !formClusters(
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          x_pos_layers,
          hitCandidateIndices,
          ut_hits,
          layer_offsets,
          n_hits_layers,
          helper,
          dev_ut_dxDy,
          true )){
      
      // go through UT layers in backward direction
      formClusters(
        hitCandidatesInLayers,
        n_hitCandidatesInLayers,
        x_pos_layers,
        hitCandidateIndices,
        ut_hits,
        layer_offsets,
        n_hits_layers,
        helper,
        dev_ut_dxDy,
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
        ut_hits,
        layer_offsets,
        n_hits_layers,
        x_pos_layers,
        hitCandidateIndices,
        veloUT_tracks_event,
        n_veloUT_tracks_event,
        bdlTable);
    }
    
  } // velo tracks
 
}
