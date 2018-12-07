#include "CompassUT.cuh"

#include "CalculateWindows.cuh"
#include "BinarySearchFirstCandidate.cuh"
#include "BinarySearch.cuh"

__global__ void compass_ut(
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
  const float* dev_unique_sector_xs, // list of xs that define the groups
  VeloUTTracking::TrackUT* dev_compassUT_tracks,
  int* dev_atomics_compassUT, // size of number of events
  int* dev_windows_layers)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[N_LAYERS];
  const uint total_number_of_hits = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks{
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states{dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  UTHitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  UTHits ut_hits {dev_ut_hits, total_number_of_hits};

  // active track pointer
  int* active_tracks = dev_active_tracks + event_number;

  // dev_atomics_compassUT contains in an SoA:
  //   1. # of veloUT tracks
  //   2. # velo tracks in UT acceptance
  // This is to write the final track
  int* n_veloUT_tracks_event = dev_atomics_compassUT + event_number;
  VeloUTTracking::TrackUT* veloUT_tracks_event = dev_compassUT_tracks + event_number * VeloUTTracking::max_num_tracks;

  // initialize atomic veloUT tracks counter && active track
  if (threadIdx.x == 0) {
    *n_veloUT_tracks_event = 0;
    *active_tracks         = 0;
  }

  __syncthreads();

  // store the tracks with valid windows
  __shared__ int shared_active_tracks[2 * VeloUTTracking::num_threads - 1];

  // store windows and num candidates in shared mem
  // 32 * 4 * 3(num_windows) * 2 (from, size) = 768 (3072 bytes)
  __shared__ int win_size_shared[VeloUTTracking::num_threads * N_LAYERS * NUM_ELEMS];

  // const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  const float* bdl_table = &(dev_ut_magnet_tool->bdlTable[0]);

  for (int i = 0; i < ((number_of_tracks_event + blockDim.x - 1) / blockDim.x) + 1; i += 1) {
    const auto i_track = i * blockDim.x + threadIdx.x;

    __syncthreads();

    if (i_track < number_of_tracks_event) {
      const uint current_track_offset = event_tracks_offset + i_track;
      // const auto velo_state = MiniState{velo_states, current_track_offset};
      
      if (!velo_states.backward[current_track_offset] && 
          // velo_track_in_UTA_acceptance(velo_state) &&
          found_active_windows(dev_windows_layers, current_track_offset) ) {
            int current_track = atomicAdd(active_tracks, 1);
            shared_active_tracks[current_track] = i_track;
      }
    }

    __syncthreads();

    const uint current_track_offset = event_tracks_offset + i_track;

    if (*active_tracks >= blockDim.x) {

      compass_ut_tracking(
        dev_windows_layers,
        dev_velo_track_hits,
        shared_active_tracks[threadIdx.x],
        event_tracks_offset + shared_active_tracks[threadIdx.x],
        velo_states,
        velo_tracks,
        ut_hits,
        ut_hit_offsets,
        bdl_table,
        dev_ut_dxDy,
        win_size_shared,
        n_veloUT_tracks_event,
        veloUT_tracks_event);

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

  __syncthreads();

  // remaining tracks
  if (threadIdx.x < *active_tracks) {

    const int i_track = shared_active_tracks[threadIdx.x];
    const uint current_track_offset = event_tracks_offset + i_track;

    compass_ut_tracking(
      dev_windows_layers,
      dev_velo_track_hits,
      i_track,
      current_track_offset,
      velo_states,
      velo_tracks,
      ut_hits,
      ut_hit_offsets,
      bdl_table,
      dev_ut_dxDy,
      win_size_shared,
      n_veloUT_tracks_event,
      veloUT_tracks_event);
  }
}

__device__ void compass_ut_tracking(
  const int* dev_windows_layers,
  uint* dev_velo_track_hits,
  const int i_track,
  const uint current_track_offset,
  const Velo::Consolidated::States& velo_states,
  const Velo::Consolidated::Tracks& velo_tracks,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_offsets,
  const float* bdl_table,
  const float* dev_ut_dxDy,
  int* win_size_shared,
  int* n_veloUT_tracks_event,
  VeloUTTracking::TrackUT* veloUT_tracks_event)
{

  // select velo track to join with UT hits
  const MiniState velo_state{velo_states, current_track_offset};

  fill_shared_windows( 
    dev_windows_layers, 
    current_track_offset, 
    win_size_shared);
  
  // Find compatible hits in the windows for this VELO track
  const auto best_hits_and_params = find_best_hits(
    win_size_shared,
    ut_hits,
    ut_hit_offsets,
    velo_state,
    dev_ut_dxDy);

  const int best_hits[N_LAYERS] = {
    std::get<0>(best_hits_and_params),
    std::get<1>(best_hits_and_params),
    std::get<2>(best_hits_and_params),
    std::get<3>(best_hits_and_params)
  };
  const BestParams best_params = std::get<4>(best_hits_and_params);

  // write the final track
  if (best_params.n_hits >= 3) {
    save_track(
      i_track,
      bdl_table,
      velo_state,
      best_params,
      dev_velo_track_hits,
      velo_tracks,
      best_params.n_hits,
      best_hits,
      ut_hits,
      dev_ut_dxDy,
      n_veloUT_tracks_event,
      veloUT_tracks_event);
  }  
}

//=============================================================================
// Fill windows and sizes for shared memory
// we store the initial hit of the window and the size of the window 
// (3 windows per layer)
//=============================================================================
__device__ __inline__ void fill_shared_windows(
  const int* windows_layers,
  const uint current_track_offset,
  int* win_size_shared)
{
  const int total_offset = NUM_ELEMS * N_LAYERS * current_track_offset;
  const int idx = NUM_ELEMS * N_LAYERS * threadIdx.x;

  // layer 0
  win_size_shared[idx]     = windows_layers[total_offset];
  win_size_shared[idx + 1] = windows_layers[total_offset + 1] - win_size_shared[idx];
  win_size_shared[idx + 2] = windows_layers[total_offset + 2];
  win_size_shared[idx + 3] = windows_layers[total_offset + 3] - win_size_shared[idx + 2];
  win_size_shared[idx + 4] = windows_layers[total_offset + 4];
  win_size_shared[idx + 5] = windows_layers[total_offset + 5] - win_size_shared[idx + 4];
  win_size_shared[idx + 6] = windows_layers[total_offset + 6];
  win_size_shared[idx + 7] = windows_layers[total_offset + 7] - win_size_shared[idx + 6];
  win_size_shared[idx + 8] = windows_layers[total_offset + 8];
  win_size_shared[idx + 9] = windows_layers[total_offset + 9] - win_size_shared[idx + 8];

  // layer 1
  win_size_shared[idx + NUM_ELEMS]     = windows_layers[total_offset + NUM_ELEMS];
  win_size_shared[idx + NUM_ELEMS + 1] = windows_layers[total_offset + NUM_ELEMS + 1] - win_size_shared[idx + NUM_ELEMS];
  win_size_shared[idx + NUM_ELEMS + 2] = windows_layers[total_offset + NUM_ELEMS + 2];
  win_size_shared[idx + NUM_ELEMS + 3] = windows_layers[total_offset + NUM_ELEMS + 3] - win_size_shared[idx + NUM_ELEMS + 2];
  win_size_shared[idx + NUM_ELEMS + 4] = windows_layers[total_offset + NUM_ELEMS + 4];
  win_size_shared[idx + NUM_ELEMS + 5] = windows_layers[total_offset + NUM_ELEMS + 5] - win_size_shared[idx + NUM_ELEMS + 4];
  win_size_shared[idx + NUM_ELEMS + 6] = windows_layers[total_offset + NUM_ELEMS + 6];
  win_size_shared[idx + NUM_ELEMS + 7] = windows_layers[total_offset + NUM_ELEMS + 7] - win_size_shared[idx + NUM_ELEMS + 6];
  win_size_shared[idx + NUM_ELEMS + 8] = windows_layers[total_offset + NUM_ELEMS + 8];
  win_size_shared[idx + NUM_ELEMS + 9] = windows_layers[total_offset + NUM_ELEMS + 9] - win_size_shared[idx + NUM_ELEMS + 8];

  // layer 2
  win_size_shared[idx + (NUM_ELEMS*2)]     = windows_layers[total_offset + (NUM_ELEMS*2)];
  win_size_shared[idx + (NUM_ELEMS*2) + 1] = windows_layers[total_offset + (NUM_ELEMS*2) + 1] - win_size_shared[idx + (NUM_ELEMS*2)];
  win_size_shared[idx + (NUM_ELEMS*2) + 2] = windows_layers[total_offset + (NUM_ELEMS*2) + 2];
  win_size_shared[idx + (NUM_ELEMS*2) + 3] = windows_layers[total_offset + (NUM_ELEMS*2) + 3] - win_size_shared[idx + (NUM_ELEMS*2) + 2];
  win_size_shared[idx + (NUM_ELEMS*2) + 4] = windows_layers[total_offset + (NUM_ELEMS*2) + 4];
  win_size_shared[idx + (NUM_ELEMS*2) + 5] = windows_layers[total_offset + (NUM_ELEMS*2) + 5] - win_size_shared[idx + (NUM_ELEMS*2) + 4];
  win_size_shared[idx + (NUM_ELEMS*2) + 6] = windows_layers[total_offset + (NUM_ELEMS*2) + 6];
  win_size_shared[idx + (NUM_ELEMS*2) + 7] = windows_layers[total_offset + (NUM_ELEMS*2) + 7] - win_size_shared[idx + (NUM_ELEMS*2) + 6];
  win_size_shared[idx + (NUM_ELEMS*2) + 8] = windows_layers[total_offset + (NUM_ELEMS*2) + 8];
  win_size_shared[idx + (NUM_ELEMS*2) + 9] = windows_layers[total_offset + (NUM_ELEMS*2) + 9] - win_size_shared[idx + (NUM_ELEMS*2) + 8];

  // layer 3
  win_size_shared[idx + (NUM_ELEMS*3)]     = windows_layers[total_offset + (NUM_ELEMS*3)];
  win_size_shared[idx + (NUM_ELEMS*3) + 1] = windows_layers[total_offset + (NUM_ELEMS*3) + 1] - win_size_shared[idx + (NUM_ELEMS*3)];
  win_size_shared[idx + (NUM_ELEMS*3) + 2] = windows_layers[total_offset + (NUM_ELEMS*3) + 2];
  win_size_shared[idx + (NUM_ELEMS*3) + 3] = windows_layers[total_offset + (NUM_ELEMS*3) + 3] - win_size_shared[idx + (NUM_ELEMS*3) + 2];
  win_size_shared[idx + (NUM_ELEMS*3) + 4] = windows_layers[total_offset + (NUM_ELEMS*3) + 4];
  win_size_shared[idx + (NUM_ELEMS*3) + 5] = windows_layers[total_offset + (NUM_ELEMS*3) + 5] - win_size_shared[idx + (NUM_ELEMS*3) + 4];
  win_size_shared[idx + (NUM_ELEMS*3) + 6] = windows_layers[total_offset + (NUM_ELEMS*3) + 6];
  win_size_shared[idx + (NUM_ELEMS*3) + 7] = windows_layers[total_offset + (NUM_ELEMS*3) + 7] - win_size_shared[idx + (NUM_ELEMS*3) + 6];
  win_size_shared[idx + (NUM_ELEMS*3) + 8] = windows_layers[total_offset + (NUM_ELEMS*3) + 8];
  win_size_shared[idx + (NUM_ELEMS*3) + 9] = windows_layers[total_offset + (NUM_ELEMS*3) + 9] - win_size_shared[idx + (NUM_ELEMS*3) + 8];
}

//=========================================================================
// Determine if there are valid windows for this track
//=========================================================================
__device__ __inline__ bool found_active_windows(
  const int* windows_layers,
  const uint current_track_offset)
{
  const int total_offset = NUM_ELEMS * N_LAYERS * current_track_offset;

  const bool main_win_found = windows_layers[total_offset]     != -1 || 
                              windows_layers[total_offset + 2] != -1 ||
                              windows_layers[total_offset + 4] != -1 ||
                              windows_layers[total_offset + 6] != -1 ||
                              windows_layers[total_offset + 8] != -1;

  const bool left_win_found = windows_layers[total_offset + (NUM_ELEMS*2)    ] != -1 ||
                              windows_layers[total_offset + (NUM_ELEMS*2) + 2] != -1 ||
                              windows_layers[total_offset + (NUM_ELEMS*2) + 4] != -1 ||
                              windows_layers[total_offset + (NUM_ELEMS*2) + 6] != -1 ||
                              windows_layers[total_offset + (NUM_ELEMS*2) + 8] != -1;

  const bool right_win_found = windows_layers[total_offset + NUM_ELEMS]         != -1 || 
                               windows_layers[total_offset + NUM_ELEMS + 2]     != -1 ||
                               windows_layers[total_offset + NUM_ELEMS + 4]     != -1 || 
                               windows_layers[total_offset + NUM_ELEMS + 6]     != -1 || 
                               windows_layers[total_offset + NUM_ELEMS + 8]     != -1 || 
                               windows_layers[total_offset + (NUM_ELEMS*3)]     != -1 ||
                               windows_layers[total_offset + (NUM_ELEMS*3) + 2] != -1 ||
                               windows_layers[total_offset + (NUM_ELEMS*3) + 4] != -1 ||
                               windows_layers[total_offset + (NUM_ELEMS*3) + 6] != -1 ||
                               windows_layers[total_offset + (NUM_ELEMS*3) + 8] != -1;

  return main_win_found && left_win_found && right_win_found;
}

// These things are all hardcopied from the PrTableForFunction and PrUTMagnetTool
// If the granularity or whatever changes, this will give wrong results
__host__ __device__ __inline__ int master_index(const int index1, const int index2, const int index3)
{
  return (index3 * 11 + index2) * 31 + index1;
}

//=========================================================================
// prepare the final track
//=========================================================================
__device__ void save_track(
  const int i_track,
  const float* bdl_table,
  const MiniState& velo_state,
  const BestParams& best_params,
  uint* dev_velo_track_hits,
  const Velo::Consolidated::Tracks& velo_tracks,
  const int num_best_hits,
  const int* best_hits,
  const UTHits& ut_hits,
  const float* ut_dxDy,
  int* n_veloUT_tracks, // increment number of tracks
  VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks]) // write the track
{
  //== Handle states. copy Velo one, add UT.
  const float zOrigin = (std::fabs(velo_state.ty) > 0.001f) ? velo_state.z - velo_state.y / velo_state.ty
                                                           : velo_state.z - velo_state.x / velo_state.tx;

  // -- These are calculations, copied and simplified from PrTableForFunction
  const float var[3] = {velo_state.ty, zOrigin, velo_state.z};

  const int index1 = std::max(0, std::min(30, int((var[0] + 0.3f) / 0.6f * 30)));
  const int index2 = std::max(0, std::min(10, int((var[1] + 250) / 500 * 10)));
  const int index3 = std::max(0, std::min(10, int(var[2] / 800 * 10)));

  assert(master_index(index1, index2, index3) < PrUTMagnetTool::N_bdl_vals);
  float bdl = bdl_table[master_index(index1, index2, index3)];

  const int num_idx = 3;
  const float bdls[num_idx] = {bdl_table[master_index(index1 + 1, index2, index3)],
                               bdl_table[master_index(index1, index2 + 1, index3)],
                               bdl_table[master_index(index1, index2, index3 + 1)]};
  const float deltaBdl[num_idx] = {0.02f, 50.0f, 80.0f};
  const float boundaries[num_idx] = {
    -0.3f + float(index1) * deltaBdl[0], -250.0f + float(index2) * deltaBdl[1], 0.0f + float(index3) * deltaBdl[2]};

  // This is an interpolation, to get a bit more precision
  float addBdlVal = 0.0f;
  const float minValsBdl[num_idx] = {-0.3f, -250.0f, 0.0f};
  const float maxValsBdl[num_idx] = {0.3f, 250.0f, 800.0f};
  for (int i = 0; i < num_idx; ++i) {
    if (var[i] < minValsBdl[i] || var[i] > maxValsBdl[i]) continue;
    const float dTab_dVar = (bdls[i] - bdl) / deltaBdl[i];
    const float dVar = (var[i] - boundaries[i]);
    addBdlVal += dTab_dVar * dVar;
  }
  bdl += addBdlVal;

  const float qpxz2p = -1 * std::sqrt(1.0f + velo_state.ty * velo_state.ty) / bdl * 3.3356f / Gaudi::Units::GeV;
  const float qop = (std::abs(bdl) < 1.e-8f) ? 0.0f : best_params.qp * qpxz2p;

  // -- Don't make tracks that have grossly too low momentum
  // -- Beware of the momentum resolution!
  const float p = 1.3f * std::abs(1 / qop);
  const float pt = p * std::sqrt(velo_state.tx * velo_state.tx + velo_state.ty * velo_state.ty);

  if (p < VeloUTConst::minMomentum || pt < VeloUTConst::minPT) return;

  // the track will be added
  int n_tracks = atomicAdd(n_veloUT_tracks, 1);

  VeloUTTracking::TrackUT track;
  track.velo_track_index = i_track;
  track.qop = qop;
  
  // Adding hits to track
  for (int i = 0; i < N_LAYERS; ++i) {
    const int hit_index = best_hits[i];
    if (hit_index >= 0) {
      track.lhcb_ids[track.number_of_hits++] = ut_hits.LHCbID[hit_index];
    }
  }

  assert(n_tracks < VeloUTTracking::max_num_tracks);
  VeloUT_tracks[n_tracks] = track;
}