#include "VeloUT.cuh"

__global__ void veloUT(
  uint* dev_ut_hits,
  uint* dev_ut_hit_count,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  const VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  const VeloState* dev_velo_states,
  VeloUTTracking::TrackUT* dev_veloUT_tracks,
  int* dev_atomics_veloUT,
  const PrUTMagnetTool* dev_ut_magnet_tool,
  float* dev_ut_dxDy
) {
  const int number_of_events = gridDim.x;
  const int event_number = blockIdx.x;
  
  const int number_of_tracks_event = *(dev_atomics_storage + event_number);
  const int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  const int accumulated_tracks_event = accumulated_tracks_base_pointer[event_number];
  const VeloState* velo_states_event = dev_velo_states + accumulated_tracks_event;
  const uint total_number_of_hits = dev_ut_hit_count[number_of_events * VeloUTTracking::n_layers];

  UTHitCount ut_hit_count;
  ut_hit_count.typecast_after_prefix_sum(dev_ut_hit_count, event_number, number_of_events);

  // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   printf("n_hits_layers[0]: %d\n", *(ut_hit_count.n_hits_layers));
  //   printf("n_hits_layers[1]: %d\n", *(ut_hit_count.n_hits_layers + 1));
  //   printf("n_hits_layers[2]: %d\n", *(ut_hit_count.n_hits_layers + 2));
  //   printf("n_hits_layers[3]: %d\n", *(ut_hit_count.n_hits_layers + 3));
  // }

  // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   printf(">>>>>>>>>>>>>>>>>>>>>>>>>>\n");
  //   printf("dev_ut_hits: %d\n", dev_ut_hits);
  //   printf("dev_ut_hit_count: %d\n", dev_ut_hit_count);
  //   printf("dev_atomics_storage: %d\n", dev_atomics_storage);
  //   printf("dev_velo_track_hit_number: %d\n", dev_velo_track_hit_number);
  //   printf("------------------\n");
  //   printf("number_of_tracks_event: %d\n", number_of_tracks_event);
  //   printf("accumulated_tracks_base_pointer: %d\n", accumulated_tracks_base_pointer);
  //   printf("accumulated_tracks_event: %d\n", accumulated_tracks_event);
  //   printf("total_number_of_hits: %d\n", total_number_of_hits);
  //   printf("<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
  // }

  UTHits ut_hits;
  ut_hits.typecast_sorted(dev_ut_hits, total_number_of_hits);
  // VeloUTTracking::HitsSoA* hits_layers_event = dev_ut_hits + event_number;

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

  __shared__ int posLayers[4][85];
         
  fillIterators(ut_hits, ut_hit_count, posLayers);

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
          ut_hit_count,
          fudgeFactors,
          velo_states_event[i_track],
          dev_ut_dxDy )
        ) continue;

    // if( !getHitsNoPosLayers(
    //       hitCandidatesInLayers,
    //       n_hitCandidatesInLayers,
    //       x_pos_layers,
    //       ut_hits,
    //       ut_hit_count,
    //       fudgeFactors,
    //       velo_states_event[i_track],
    //       dev_ut_dxDy )
    //     ) continue;

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
          ut_hit_count,
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
        ut_hit_count,
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
        ut_hit_count,
        x_pos_layers,
        hitCandidateIndices,
        veloUT_tracks_event,
        n_veloUT_tracks_event,
        bdlTable);
    }
    
  } // velo tracks
 
}
