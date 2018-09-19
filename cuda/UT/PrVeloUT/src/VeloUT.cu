#include "VeloUT.cuh"

__global__ void veloUT(
  uint* dev_ut_hits, // actual hit content
  uint* dev_ut_hit_count, // prefixsum, count per layer
  int* dev_atomics_storage, // semi_prefixsum, offset to tracks
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  VeloUTTracking::TrackUT* dev_veloUT_tracks,
  int* dev_atomics_veloUT,
  PrUTMagnetTool* dev_ut_magnet_tool,
  float* dev_ut_dxDy,
  int* dev_active_tracks) // size of number of events
{
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

  UTHits ut_hits;
  ut_hits.typecast_sorted(dev_ut_hits, total_number_of_hits);

  // active track pointer
  int* active_tracks = dev_active_tracks + event_number;

  // dev_atomics_veloUT contains in an SoA:
  //   1. # of veloUT tracks
  //   2. # velo tracks in UT acceptance
  // This is to write the final track
  int* n_veloUT_tracks_event = dev_atomics_veloUT + event_number;
  VeloUTTracking::TrackUT* veloUT_tracks_event = dev_veloUT_tracks + event_number * VeloUTTracking::max_num_tracks;
  
  // initialize atomic veloUT tracks counter && active track
  if ( threadIdx.x == 0 ) {
    *n_veloUT_tracks_event = 0;
    *active_tracks = 0;
  }

  __shared__ int shared_active_tracks[2 * VeloUTTracking::num_threads - 1];

  __shared__ int posLayers[VeloUTTracking::n_layers][VeloUTTracking::n_iterations_pos];

  __syncthreads();
         
  fillIterators(ut_hits, ut_hit_count, posLayers);

  const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  const float* bdlTable     = &(dev_ut_magnet_tool->bdlTable[0]);

  // array to store indices of selected hits in layers
  // -> can then access the hit information in the HitsSoA
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers];
  
  for ( int i = threadIdx.x; i < number_of_tracks_event; i+=blockDim.x) {

    // __syncthreads();

    const uint velo_states_index = event_tracks_offset + i;
    if (!velo_states.backward[velo_states_index]) {
      // Using Mini State with only x, y, tx, ty and z
      if(veloTrackInUTAcceptance(MiniState{velo_states, velo_states_index})) {
        int current_active_track = atomicAdd(active_tracks, 1);
        shared_active_tracks[current_active_track] = i;
      }
    }

    __syncthreads();

    if (*active_tracks >= VeloUTTracking::num_threads) {

      const int i_track = shared_active_tracks[threadIdx.x];

      // for storing calculated x position of hits for this track
      float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];

      if (process_track(
        i_track,
        event_tracks_offset,
        velo_states,
        hitCandidatesInLayers,
        n_hitCandidatesInLayers,
        x_pos_layers,
        posLayers,
        ut_hits,
        ut_hit_count,
        fudgeFactors,
        dev_ut_dxDy)
      ) {
          process_track2(
          i_track,
          event_tracks_offset,
          velo_states,
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          x_pos_layers,
          ut_hits,
          ut_hit_count,
          dev_velo_track_hits,
          velo_tracks,
          n_veloUT_tracks_event,
          veloUT_tracks_event,
          bdlTable,
          dev_ut_dxDy);    
      }

      __syncthreads();

      const int j = blockDim.x + threadIdx.x;
      if (j < *active_tracks) {
        shared_active_tracks[threadIdx.x] = shared_active_tracks[j];
      }

      __syncthreads();

      if (threadIdx.x == 0) {
        *active_tracks -= blockDim.x;
      }
      
    } 
  }

  // remaining tracks 
  if (threadIdx.x < *active_tracks) {

    const int i_track = shared_active_tracks[threadIdx.x];

    // for storing calculated x position of hits for this track
    float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];

    if (process_track(
      i_track,
      event_tracks_offset,
      velo_states,
      hitCandidatesInLayers,
      n_hitCandidatesInLayers,
      x_pos_layers,
      posLayers,
      ut_hits,
      ut_hit_count,
      fudgeFactors,
      dev_ut_dxDy)
    ) {
        process_track2(
        i_track,
        event_tracks_offset,
        velo_states,
        hitCandidatesInLayers,
        n_hitCandidatesInLayers,
        x_pos_layers,
        ut_hits,
        ut_hit_count,
        dev_velo_track_hits,
        velo_tracks,
        n_veloUT_tracks_event,
        veloUT_tracks_event,
        bdlTable,
        dev_ut_dxDy);    
    }
  }
}

// vertical processing
__device__ bool process_track(
  const int i_track,
  const uint event_tracks_offset,
  const Velo::Consolidated::States& velo_states,
  int (&hitCandidatesInLayers)[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int (&n_hitCandidatesInLayers)[VeloUTTracking::n_layers],
  float (&x_pos_layers)[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int (&posLayers)[VeloUTTracking::n_layers][VeloUTTracking::n_iterations_pos],
  UTHits& ut_hits,
  UTHitCount& ut_hit_count,
  const float* fudgeFactors,
  float* dev_ut_dxDy
) {
  // MiniState aux_velo_state {velo_states, velo_states_index};
  const uint velo_states_index = event_tracks_offset + i_track;
  const MiniState velo_state {velo_states, velo_states_index};

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
      ) { return false; }

  // if( (layer == 3 || layer == 4) && nLayers == 0) return false;
  // if( layer == 4 && nLayers < 2 ) return false;

  // // there are hits if at least nLayers was 2 (so 2 layers had hits)
  // return nLayers > 2;

  return true;
}

// horizontal processing
__device__ void process_track2 (
  const int i_track,
  const uint event_tracks_offset,
  const Velo::Consolidated::States& velo_states,
  int (&hitCandidatesInLayers)[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int (&n_hitCandidatesInLayers)[VeloUTTracking::n_layers],
  float (&x_pos_layers)[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  UTHits& ut_hits,
  UTHitCount& ut_hit_count,
  uint* dev_velo_track_hits,
  const Velo::Consolidated::Tracks& velo_tracks,
  int* n_veloUT_tracks_event,
  VeloUTTracking::TrackUT* veloUT_tracks_event,
  const float* bdlTable,
  float* dev_ut_dxDy
) {
  // MiniState aux_velo_state {velo_states, velo_states_index};
  const uint velo_states_index = event_tracks_offset + i_track;
  MiniState velo_state {velo_states, velo_states_index};

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
}
