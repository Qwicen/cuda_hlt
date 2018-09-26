#include "compassUT.cuh"

__global__ void compassUT(
  uint* dev_ut_hits, // actual hit content
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage, // semi_prefixsum, offset to tracks
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  int* dev_active_tracks,
  const uint* dev_unique_x_sector_layer_offsets, // prefixsum to point to the x hit of the sector, per layer
  const uint* dev_unique_x_sector_offsets, // TODO remove this, only needed for decoding
  const float* dev_unique_sector_xs, // list of xs that define the groups
  VeloUTTracking::TrackUT* dev_compassUT_tracks,
  int* dev_atomics_compassUT, // size of number of events
  int* dev_windows_layers)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint total_number_of_hits = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];
  
  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks {(uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states {dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  UTHitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  
  UTHits ut_hits;
  ut_hits.typecast_sorted(dev_ut_hits, total_number_of_hits);

  // active track pointer
  int* active_tracks = dev_active_tracks + event_number;

  // dev_atomics_compassUT contains in an SoA:
  //   1. # of veloUT tracks
  //   2. # velo tracks in UT acceptance
  // This is to write the final track
  int* n_veloUT_tracks_event = dev_atomics_compassUT + event_number;
  VeloUTTracking::TrackUT* veloUT_tracks_event = dev_compassUT_tracks + event_number * VeloUTTracking::max_num_tracks;
  
  // initialize atomic veloUT tracks counter && active track
  if ( threadIdx.x == 0 ) {
    *n_veloUT_tracks_event = 0;
    *active_tracks = 0;
  }

  int shared_active_tracks[2 * VeloUTTracking::num_threads - 1];

  // __syncthreads();

  const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  const float* bdlTable     = &(dev_ut_magnet_tool->bdlTable[0]);

  // array to store indices of selected hits in layers
  // -> can then access the hit information in the HitsSoA
  // int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
  // int n_hitCandidatesInLayers[VeloUTTracking::n_layers];
  
  for ( int i = threadIdx.x; i < number_of_tracks_event; i+=blockDim.x) {

    // // __syncthreads();

    // const uint velo_states_index = event_tracks_offset + i;
    // if (!velo_states.backward[velo_states_index]) {
    //   // Using Mini State with only x, y, tx, ty and z
    //   if(velo_track_in_UTA_acceptance(MiniState{velo_states, velo_states_index})) {
    //     int current_active_track = atomicAdd(active_tracks, 1);
    //     shared_active_tracks[current_active_track] = i;
    //   }
    // }

    // // TODO the non active tracks should be -1 

    // __syncthreads();

    // if (*active_tracks >= VeloUTTracking::num_threads) {

    //   const int i_track = shared_active_tracks[threadIdx.x];

    //   // for storing calculated x position of hits for this track
    //   // float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];

    //   // store a window(2 positions) for each layer, for each thrack
    //   int windows_layers[VeloUTTracking::num_threads * VeloUTTracking::n_layers * 2];
    //   // TODO change to num_tracks
    //   // TODO move to global

    //   // MiniState aux_velo_state {velo_states, velo_states_index};
    //   const uint velo_states_index = event_tracks_offset + i_track;
    //   const MiniState velo_state {velo_states, velo_states_index};

    //   get_windows(
    //     i_track,
    //     velo_state,
    //     fudgeFactors,
    //     ut_hits,
    //     ut_hit_offsets,
    //     dev_ut_dxDy,
    //     dev_unique_sector_xs,
    //     // dev_unique_x_sector_offsets,
    //     dev_unique_x_sector_layer_offsets,
    //     velo_tracks,
    //     (int*) &windows_layers[0]);

    //   __syncthreads();

    //   // float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];

    //   TrackHelper helper {velo_state};

    //   // indices within hitCandidatesInLayers for selected hits belonging to best track 
    //   float x_hit_layer[VeloUTTracking::n_layers];
    //   int hitCandidateIndices[VeloUTTracking::n_layers];

    //   // go through UT layers in forward direction
    //   if(!find_best_hits(
    //         i_track,
    //         (int*) &windows_layers[0],
    //         ut_hits,
    //         ut_hit_count,
    //         velo_state,
    //         dev_ut_dxDy,
    //         true,
    //         helper,
    //         x_hit_layer,
    //         hitCandidateIndices)) {
        
    //     // go through UT layers in backward direction
    //     find_best_hits(
    //         i_track,
    //         (int*) &windows_layers[0],
    //         ut_hits,
    //         ut_hit_count,
    //         velo_state,
    //         dev_ut_dxDy,
    //         false,
    //         helper,
    //         x_hit_layer,
    //         hitCandidateIndices);
    //   }

  //     if ( helper.n_hits > 0 ) {
  //       const uint velo_track_hit_number = velo_tracks.number_of_hits(i_track);
  //       const Velo::Consolidated::Hits velo_track_hits = velo_tracks.get_hits(dev_velo_track_hits, i_track);

  //       prepareOutputTrack(
  //         i_track,
  //         velo_track_hits,
  //         velo_track_hit_number,
  //         helper,
  //         velo_state,
  //         (int*) &windows_layers[0],
  //         ut_hits,
  //         ut_hit_count,
  //         (float*) &x_hit_layer[0],
  //         (int*) &hitCandidateIndices[0],
  //         bdlTable,
  //         veloUT_tracks_event,
  //         n_veloUT_tracks_event);
  //     }
      
  //     const int j = blockDim.x + threadIdx.x;
  //     if (j < *active_tracks) {
  //       shared_active_tracks[threadIdx.x] = shared_active_tracks[j];
  //     }

  //     __syncthreads();

  //     if (threadIdx.x == 0) {
  //       *active_tracks -= blockDim.x;
  //     }
      
  //   } 
  }

  // // remaining tracks 
  // if (threadIdx.x < *active_tracks) {

  //   // store a window(2 positions) for each layer, for each thrack
  //   __shared__ int windows_layers[VeloUTTracking::num_threads * VeloUTTracking::n_layers * 2];

  //   const int i_track = shared_active_tracks[threadIdx.x];

  //   // MiniState aux_velo_state {velo_states, velo_states_index};
  //   const uint velo_states_index = event_tracks_offset + i_track;
  //   const MiniState velo_state {velo_states, velo_states_index};

  //   get_windows(
  //     i_track,
  //     velo_state,
  //     fudgeFactors,
  //     ut_hits,
  //     ut_hit_count,
  //     dev_ut_dxDy,
  //     (int*) &windows_layers[0]);

  //   __syncthreads();

  //   // for storing calculated x position of hits for this track
  //   // float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];

  // }
}