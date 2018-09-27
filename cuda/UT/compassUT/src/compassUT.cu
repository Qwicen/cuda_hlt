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
  const uint event_number     = blockIdx.x;

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint total_number_of_hits       = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks{
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states{dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset    = velo_tracks.tracks_offset(event_number);

  UTHitOffsets ut_hit_offsets{
    dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};

  UTHits ut_hits;
  ut_hits.typecast_sorted(dev_ut_hits, total_number_of_hits);

  // active track pointer
  // int* active_tracks = dev_active_tracks + event_number;

  // dev_atomics_compassUT contains in an SoA:
  //   1. # of veloUT tracks
  //   2. # velo tracks in UT acceptance
  // This is to write the final track
  int* n_veloUT_tracks_event                   = dev_atomics_compassUT + event_number;
  // VeloUTTracking::TrackUT* veloUT_tracks_event = dev_compassUT_tracks + event_number * VeloUTTracking::max_num_tracks;

  // initialize atomic veloUT tracks counter && active track
  if (threadIdx.x == 0) {
    *n_veloUT_tracks_event = 0;
    // *active_tracks         = 0;
  }

  // int shared_active_tracks[2 * VeloUTTracking::num_threads - 1];

  // __syncthreads();

  // const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  // const float* bdlTable     = &(dev_ut_magnet_tool->bdlTable[0]);

  // array to store indices of selected hits in layers
  // -> can then access the hit information in the HitsSoA
  // int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
  // int n_hitCandidatesInLayers[VeloUTTracking::n_layers];

  for (int i = threadIdx.x; i < number_of_tracks_event; i += blockDim.x) {

    // __syncthreads();

    // TODO the non active tracks should be -1
    // const int i_track = shared_active_tracks[threadIdx.x];

    const uint velo_states_index = event_tracks_offset + i;
    const MiniState velo_state{velo_states, velo_states_index};

    //   __syncthreads();

    TrackHelper helper{velo_state};

    // indices within hitCandidatesInLayers for selected hits belonging to best track
    float x_hit_layer[VeloUTTracking::n_layers];
    int hitCandidateIndices[VeloUTTracking::n_layers];

    // go through UT layers in forward direction
    if (!find_best_hits(
          i,
          dev_windows_layers,
          ut_hits,
          ut_hit_offsets,
          velo_state,
          dev_ut_dxDy,
          true,
          helper,
          x_hit_layer,
          hitCandidateIndices)) {

      // go through UT layers in backward direction
      find_best_hits(
        i,
        dev_windows_layers,
        ut_hits,
        ut_hit_offsets,
        velo_state,
        dev_ut_dxDy,
        false,
        helper,
        x_hit_layer,
        hitCandidateIndices);
    }

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

//=========================================================================
// hits_to_track
//=========================================================================
__host__ __device__ bool find_best_hits(
  const int i_track,
  const int* dev_windows_layers,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const bool forward,
  TrackHelper& helper,
  float* x_hit_layer,
  int* bestHitCandidateIndices)
{
  // handle forward / backward cluster search
  int layers[N_LAYERS];
  for (int i_layer = 0; i_layer < N_LAYERS; ++i_layer) {
    if (forward)
      layers[i_layer] = i_layer;
    else
      layers[i_layer] = N_LAYERS - 1 - i_layer;
  }

  bool fourLayerSolution = false;
  int hitCandidateIndices[N_LAYERS];
  const float yyProto = velo_state.y - velo_state.ty * velo_state.z;

  // Get windows of all layers
  const auto candidates = const_cast<track_candidates*>(
    reinterpret_cast<const track_candidates*>(dev_windows_layers + 2 * N_LAYERS * i_track));
  const uint from0 = dev_windows_layers[candidates->layer[0].first];
  const uint to0   = dev_windows_layers[candidates->layer[0].last];
  const uint from2 = dev_windows_layers[candidates->layer[2].first];
  const uint to2   = dev_windows_layers[candidates->layer[2].last];
  const uint from1 = dev_windows_layers[candidates->layer[1].first];
  const uint to1   = dev_windows_layers[candidates->layer[1].last];
  const uint from3 = dev_windows_layers[candidates->layer[3].first];
  const uint to3   = dev_windows_layers[candidates->layer[3].last];

  // layer 0
  for (int i_hit0 = from0; i_hit0 < to0; ++i_hit0) {

    const float yy0 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit0]);
    x_hit_layer[0]  = ut_hits.xAt(i_hit0, yy0, ut_dxDy[layers[0]]);
    // ---------------------------------------

    const float zhitLayer0 = ut_hits.zAtYEq0[i_hit0];
    hitCandidateIndices[0] = i_hit0;

    // layer 2
    for (int i_hit2 = from2; i_hit2 < to2; ++i_hit2) {
      // x_pos_layers calc
      const float yy2 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit2]);
      x_hit_layer[2]  = ut_hits.xAt(i_hit2, yy2, ut_dxDy[layers[2]]);
      // ---------------------------------------

      const float zhitLayer2 = ut_hits.zAtYEq0[i_hit2];
      hitCandidateIndices[2] = i_hit2;

      const float tx = (x_hit_layer[2] - x_hit_layer[0]) / (zhitLayer2 - zhitLayer0);
      if (std::abs(tx - velo_state.tx) > PrVeloUTConst::deltaTx2) continue;

      float hitTol         = PrVeloUTConst::hitTol2;
      int index_best_hit_1 = -1;

      // layer 1
      for (int i_hit1 = from1; i_hit1 < to1; ++i_hit1) {
        // x_pos_layers calc
        const float yy1 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit1]);
        x_hit_layer[1]  = ut_hits.xAt(i_hit1, yy1, ut_dxDy[layers[1]]);
        // ---------------------------------------
        const float zhitLayer1 = ut_hits.zAtYEq0[i_hit1];

        const float xextrapLayer1 = x_hit_layer[0] + tx * (zhitLayer1 - zhitLayer0);
        if (std::abs(x_hit_layer[1] - xextrapLayer1) < hitTol) {
          hitTol                 = std::abs(x_hit_layer[1] - xextrapLayer1);
          index_best_hit_1       = i_hit1;
          hitCandidateIndices[1] = i_hit1;
        }
      }

      if (fourLayerSolution && index_best_hit_1 < 0) continue;

      int index_best_hit_3 = -1;
      hitTol               = PrVeloUTConst::hitTol2;

      // layer 3
      for (int i_hit3 = from3; i_hit3 < to3; ++i_hit3) {
        // x_pos_layers calc
        const float yy3 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit3]);
        x_hit_layer[3]  = ut_hits.xAt(i_hit3, yy3, ut_dxDy[layers[3]]);
        // ---------------------------------------
        const float zhitLayer3 = ut_hits.zAtYEq0[i_hit3];

        const float xextrapLayer3 = x_hit_layer[2] + tx * (zhitLayer3 - zhitLayer2);
        if (std::abs(x_hit_layer[3] - xextrapLayer3) < hitTol) {
          hitTol                 = std::abs(x_hit_layer[3] - xextrapLayer3);
          index_best_hit_3       = i_hit3;
          hitCandidateIndices[3] = i_hit3;
        }
      }

      // -- All hits found
      if (index_best_hit_1 > 0 && index_best_hit_3 > 0) {
        const int hitIndices[4] = {i_hit0, index_best_hit_1, i_hit2, index_best_hit_3};
        simple_fit<4>(
          x_hit_layer, hitCandidateIndices, ut_hits, hitIndices, velo_state, ut_dxDy, bestHitCandidateIndices, helper);

        if (!fourLayerSolution && helper.n_hits > 0) { fourLayerSolution = true; }
        continue;
      }

      // -- Nothing found in layer 3
      if (!fourLayerSolution && index_best_hit_1 > 0) {
        const int hitIndices[3] = {i_hit0, index_best_hit_1, i_hit2};
        simple_fit<3>(
          x_hit_layer, hitCandidateIndices, ut_hits, hitIndices, velo_state, ut_dxDy, bestHitCandidateIndices, helper);
        continue;
      }
      // -- Nothing found in layer 1
      if (!fourLayerSolution && x_hit_layer[3] > 0) {
        hitCandidateIndices[1]  = hitCandidateIndices[3]; // hit3 saved in second position of hits4fit
        const int hitIndices[3] = {i_hit0, index_best_hit_3, i_hit2};
        simple_fit<3>(
          x_hit_layer, hitCandidateIndices, ut_hits, hitIndices, velo_state, ut_dxDy, bestHitCandidateIndices, helper);
        continue;
      }
    }
  }

  return fourLayerSolution;
}