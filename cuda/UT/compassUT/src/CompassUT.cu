#include "CompassUT.cuh"

#include "BinarySearch.cuh"
#include "BinarySearchFirstCandidate.cuh"
#include "CalculateWindows.cuh"

__global__ void compass_ut(
  uint* dev_ut_hits, // actual hit content
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage, // semi_prefixsum, offset to tracks
  uint* dev_velo_track_hit_number,
  char* dev_velo_track_hits,
  char* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  int* dev_active_tracks,
  const uint* dev_unique_x_sector_layer_offsets, // prefixsum to point to the x hit of the sector, per layer
  const float* dev_unique_sector_xs,             // list of xs that define the groups
  UT::TrackHits* dev_compassUT_tracks,
  int* dev_atomics_compassUT, // size of number of events
  short* dev_windows_layers,
  bool* dev_accepted_velo_tracks)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[UT::Constants::n_layers];
  const uint total_number_of_hits = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks {
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states {dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  short* windows_layers = dev_windows_layers + event_tracks_offset * CompassUT::num_elems * UT::Constants::n_layers;

  const UT::HitOffsets ut_hit_offsets {
    dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  const UT::Hits ut_hits {dev_ut_hits, total_number_of_hits};
  const auto event_hit_offset = ut_hit_offsets.event_offset();

  // active track pointer
  int* active_tracks = dev_active_tracks + event_number;

  // dev_atomics_compassUT contains in an SoA:
  //   1. # of veloUT tracks
  //   2. # velo tracks in UT acceptance
  // This is to write the final track
  int* n_veloUT_tracks_event = dev_atomics_compassUT + event_number;
  UT::TrackHits* veloUT_tracks_event = dev_compassUT_tracks + event_number * UT::Constants::max_num_tracks;

  // initialize atomic veloUT tracks counter && active track
  if (threadIdx.x == 0) {
    *n_veloUT_tracks_event = 0;
    *active_tracks = 0;
  }

  __syncthreads();

  // store the tracks with valid windows
  __shared__ int shared_active_tracks[2 * UT::Constants::num_thr_compassut - 1];

  // store windows and num candidates in shared mem
  __shared__ short win_size_shared[UT::Constants::num_thr_compassut * UT::Constants::n_layers * CompassUT::num_elems];

  // const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  const float* bdl_table = &(dev_ut_magnet_tool->bdlTable[0]);

  for (int i = 0; i < ((number_of_tracks_event + blockDim.x - 1) / blockDim.x) + 1; i += 1) {
    const auto i_track = i * blockDim.x + threadIdx.x;

    __syncthreads();

    if (i_track < number_of_tracks_event) {
      const uint current_track_offset = event_tracks_offset + i_track;
      const auto velo_state = MiniState {velo_states, current_track_offset};

      if (
        !velo_states.backward[current_track_offset] && dev_accepted_velo_tracks[current_track_offset] &&
        velo_track_in_UTA_acceptance(velo_state) &&
        found_active_windows(windows_layers, number_of_tracks_event, i_track)) {
        int current_track = atomicAdd(active_tracks, 1);
        shared_active_tracks[current_track] = i_track;
      }
    }

    __syncthreads();

    if (*active_tracks >= blockDim.x) {

      compass_ut_tracking(
        windows_layers,
        dev_velo_track_hits,
        number_of_tracks_event,
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
        veloUT_tracks_event,
        event_hit_offset);

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
    compass_ut_tracking(
      windows_layers,
      dev_velo_track_hits,
      number_of_tracks_event,
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
      veloUT_tracks_event,
      event_hit_offset);
  }
}

__device__ void compass_ut_tracking(
  const short* windows_layers,
  char* dev_velo_track_hits,
  const uint number_of_tracks_event,
  const int i_track,
  const uint current_track_offset,
  const Velo::Consolidated::States& velo_states,
  const Velo::Consolidated::Tracks& velo_tracks,
  const UT::Hits& ut_hits,
  const UT::HitOffsets& ut_hit_offsets,
  const float* bdl_table,
  const float* dev_ut_dxDy,
  short* win_size_shared,
  int* n_veloUT_tracks_event,
  UT::TrackHits* veloUT_tracks_event,
  const int event_hit_offset)
{

  // select velo track to join with UT hits
  const MiniState velo_state {velo_states, current_track_offset};

  fill_shared_windows(windows_layers, number_of_tracks_event, i_track, win_size_shared);

  // Find compatible hits in the windows for this VELO track
  const auto best_hits_and_params =
    find_best_hits(win_size_shared, number_of_tracks_event, i_track, ut_hits, ut_hit_offsets, velo_state, dev_ut_dxDy);

  const int best_hits[UT::Constants::n_layers] = {std::get<0>(best_hits_and_params),
                                                  std::get<1>(best_hits_and_params),
                                                  std::get<2>(best_hits_and_params),
                                                  std::get<3>(best_hits_and_params)};
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
      veloUT_tracks_event,
      event_hit_offset);
  }
}

//=============================================================================
// Fill windows and sizes for shared memory
// we store the initial hit of the window and the size of the window
// (3 windows per layer)
//=============================================================================
__device__ __inline__ void fill_shared_windows(
  const short* windows_layers,
  const int number_of_tracks_event,
  const int i_track,
  short* win_size_shared)
{
  const int track_pos = UT::Constants::n_layers * number_of_tracks_event;
  const int track_pos_sh = UT::Constants::n_layers * UT::Constants::num_thr_compassut;

  for (int layer = 0; layer < UT::Constants::n_layers; ++layer) {
    for (int pos = 0; pos < CompassUT::num_elems; ++pos) {
      win_size_shared[pos * track_pos_sh + layer * UT::Constants::num_thr_compassut + threadIdx.x] =
        windows_layers[pos * track_pos + layer * number_of_tracks_event + i_track];
    }
  }
}

//=========================================================================
// Determine if there are valid windows for this track looking at the sizes
//=========================================================================
__device__ __inline__ bool
found_active_windows(const short* windows_layers, const int number_of_tracks_event, const int i_track)
{
  const int track_pos = UT::Constants::n_layers * number_of_tracks_event;

  // The windows are stored in SOA, with the first 5 arrays being the first hits of the windows,
  // and the next 5 the sizes of the windows. We check the sizes of all the windows.
  const bool l0_found = windows_layers[5 * track_pos + 0 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[6 * track_pos + 0 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[7 * track_pos + 0 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[8 * track_pos + 0 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[9 * track_pos + 0 * number_of_tracks_event + i_track] != 0;

  const bool l1_found = windows_layers[5 * track_pos + 1 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[6 * track_pos + 1 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[7 * track_pos + 1 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[8 * track_pos + 1 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[9 * track_pos + 1 * number_of_tracks_event + i_track] != 0;

  const bool l2_found = windows_layers[5 * track_pos + 2 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[6 * track_pos + 2 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[7 * track_pos + 2 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[8 * track_pos + 2 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[9 * track_pos + 2 * number_of_tracks_event + i_track] != 0;

  const bool l3_found = windows_layers[5 * track_pos + 3 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[6 * track_pos + 3 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[7 * track_pos + 3 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[8 * track_pos + 3 * number_of_tracks_event + i_track] != 0 ||
                        windows_layers[9 * track_pos + 3 * number_of_tracks_event + i_track] != 0;

  return (l0_found && l2_found && (l1_found || l3_found)) || (l3_found && l1_found && (l2_found || l0_found));
}

//=========================================================================
// These things are all hardcopied from the PrTableForFunction and PrUTMagnetTool
// If the granularity or whatever changes, this will give wrong results
//=========================================================================
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
  char* dev_velo_track_hits,
  const Velo::Consolidated::Tracks& velo_tracks,
  const int num_best_hits,
  const int* best_hits,
  const UT::Hits& ut_hits,
  const float* ut_dxDy,
  int* n_veloUT_tracks,         // increment number of tracks
  UT::TrackHits* VeloUT_tracks, // write the track
  const int event_hit_offset)
{
  //== Handle states. copy Velo one, add UT.
  const float zOrigin = (std::fabs(velo_state.ty) > 0.001f) ? velo_state.z - velo_state.y / velo_state.ty :
                                                              velo_state.z - velo_state.x / velo_state.tx;

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

  if (p < UT::Constants::minMomentum || pt < UT::Constants::minPT) return;

  // the track will be added
  int n_tracks = atomicAdd(n_veloUT_tracks, 1);

  UT::TrackHits track;
  track.velo_track_index = i_track;
  track.qop = qop;
  track.hits_num = 0;

  // Adding hits to track
  for (int i = 0; i < UT::Constants::n_layers; ++i) {
    if (best_hits[i] != -1) {
      track.hits[i] = best_hits[i] - event_hit_offset;
      ++track.hits_num;
    }
    else {
      track.hits[i] = -1;
    }
  }

  assert(n_tracks < UT::Constants::max_num_tracks);
  VeloUT_tracks[n_tracks] = track;
}
