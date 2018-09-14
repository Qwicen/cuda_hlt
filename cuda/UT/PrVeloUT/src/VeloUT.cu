#include "VeloUT.cuh"

__global__ void veloUT(
  uint* dev_ut_hits,
  uint* dev_ut_hit_count,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  VeloUTTracking::TrackUT* dev_veloUT_tracks,
  int* dev_atomics_veloUT,
  PrUTMagnetTool* dev_ut_magnet_tool,
  float* dev_ut_dxDy
) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint total_number_of_hits = dev_ut_hit_count[number_of_events * VeloUTTracking::n_layers];
  
  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks {(uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states {dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

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
    
    const uint velo_states_index = event_tracks_offset + i_track;
    if (i_track >= number_of_tracks_event) continue;
    if (velo_states.backward[velo_states_index]) continue;

    // Mini State with only x, y, tx, ty and z
    MiniState velo_state {velo_states, velo_states_index};

    if(!veloTrackInUTAcceptance(velo_state)) continue;

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
          velo_state,
          dev_ut_dxDy)
        ) continue;

    TrackHelper helper {velo_state};

    // indices within hitCandidatesInLayers for selected hits belonging to best track 
    int hitCandidateIndices[VeloUTTracking::n_layers];
    
    // go through UT layers in forward direction
    if(!formClusters(
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          x_pos_layers,
          hitCandidateIndices,
          ut_hits,
          ut_hit_count,
          helper,
          velo_state,
          dev_ut_dxDy,
          true)) {
      
      // go through UT layers in backward direction
      formClusters(
        hitCandidatesInLayers,
        n_hitCandidatesInLayers,
        x_pos_layers,
        hitCandidateIndices,
        ut_hits,
        ut_hit_count,
        helper,
        velo_state,
        dev_ut_dxDy,
        false);
    }
    
    if ( helper.n_hits > 0 ) {
      const uint velo_track_hit_number = velo_tracks.number_of_hits(i_track);
      const Velo::Consolidated::Hits velo_track_hits = velo_tracks.get_hits(dev_velo_track_hits, i_track);

      prepareOutputTrack(
        velo_track_hits,
        velo_track_hit_number,
        helper,
        velo_state,
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
