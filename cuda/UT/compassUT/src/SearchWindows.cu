#include "CalculateWindows.cuh"
#include "SearchWindows.cuh"
#include "Handler.cuh"
#include <tuple>

__global__ void ut_search_windows(
  uint* dev_ut_hits, // actual hit content
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage, // semi_prefixsum, offset to tracks
  uint* dev_velo_track_hit_number,
  char* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  const uint* dev_unique_x_sector_layer_offsets, // prefixsum to point to the x hit of the sector, per layer
  const float* dev_unique_sector_xs,             // list of xs that define the groups
  short* dev_windows_layers,
  int* dev_active_tracks,
  bool* dev_accepted_velo_tracks)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint layer = threadIdx.x;
  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[UT::Constants::n_layers];
  const uint total_number_of_hits = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks {
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states {dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  UT::HitOffsets ut_hit_offsets {
    dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  UT::Hits ut_hits {dev_ut_hits, total_number_of_hits};

  const uint layer_offset = ut_hit_offsets.layer_offset(layer);
  const float* fudge_factors = &(dev_ut_magnet_tool->dxLayTable[0]);
  int* active_tracks = dev_active_tracks + event_number;

  // Store only the valid tracks into shared memory.
  // Fill the array until with find enough valid tracks = block size
  __shared__ int shared_active_tracks[2 * UT::Constants::num_thr_searchwin - 1];

  for (int i = 0; i < ((number_of_tracks_event + blockDim.y - 1) / blockDim.y) + 1; i += 1) {
    const auto i_track = i * blockDim.y + threadIdx.y;

    __syncthreads();

    // filter the tracks that won't be valid
    if (threadIdx.x == 0) {
      const uint current_track_offset = event_tracks_offset + i_track;
      const auto velo_state = MiniState {velo_states, current_track_offset};
      if (i_track < number_of_tracks_event) {
        if (
          !velo_states.backward[current_track_offset] && dev_accepted_velo_tracks[current_track_offset] &&
          velo_track_in_UTA_acceptance(velo_state)) {
          int current_track = atomicAdd(active_tracks, 1);
          shared_active_tracks[current_track] = i_track;
        }
      }
    }

    __syncthreads();

    // process only the active tracks
    if (*active_tracks >= blockDim.y) {

      const uint current_track_offset = event_tracks_offset + shared_active_tracks[threadIdx.y];
      const auto velo_state = MiniState {velo_states, current_track_offset};

      const auto candidates = calculate_windows(
        shared_active_tracks[threadIdx.y],
        layer,
        velo_state,
        fudge_factors,
        ut_hits,
        ut_hit_offsets,
        dev_ut_dxDy,
        dev_unique_sector_xs,
        dev_unique_x_sector_layer_offsets,
        velo_tracks);

      // Write the windows in SoA style
      short* windows_layers = dev_windows_layers + event_tracks_offset * CompassUT::num_elems * UT::Constants::n_layers;

      const int track_pos = UT::Constants::n_layers * number_of_tracks_event;
      const int layer_pos = layer * number_of_tracks_event + shared_active_tracks[threadIdx.y];

      windows_layers[0 * track_pos + layer_pos] = std::get<0>(candidates) - layer_offset; // first_candidate
      windows_layers[1 * track_pos + layer_pos] = std::get<2>(candidates) - layer_offset; // left_group_first
      windows_layers[2 * track_pos + layer_pos] = std::get<4>(candidates) - layer_offset; // right_group_first
      windows_layers[3 * track_pos + layer_pos] = std::get<6>(candidates) - layer_offset; // left2_group_first
      windows_layers[4 * track_pos + layer_pos] = std::get<8>(candidates) - layer_offset; // right2_group_first
      windows_layers[5 * track_pos + layer_pos] = std::get<1>(candidates) - std::get<0>(candidates); // last_size
      windows_layers[6 * track_pos + layer_pos] = std::get<3>(candidates) - std::get<2>(candidates); // left_size_last
      windows_layers[7 * track_pos + layer_pos] = std::get<5>(candidates) - std::get<4>(candidates); // right_size_first
      windows_layers[8 * track_pos + layer_pos] = std::get<7>(candidates) - std::get<6>(candidates); // left2_size_last
      windows_layers[9 * track_pos + layer_pos] =
        std::get<9>(candidates) - std::get<8>(candidates); // right2_size_first

      __syncthreads();

      if (threadIdx.x == 0) {
        const int j = blockDim.y + threadIdx.y;
        if (j < *active_tracks) {
          shared_active_tracks[threadIdx.y] = shared_active_tracks[j];
        }
      }

      __syncthreads();

      if (threadIdx.x == 0 && threadIdx.y == 0) {
        *active_tracks -= blockDim.y;
      }
    }
  }

  __syncthreads();

  // remaining tracks
  if (threadIdx.y < *active_tracks) {

    const int i_track = shared_active_tracks[threadIdx.y];
    const uint current_track_offset = event_tracks_offset + i_track;

    const auto velo_state = MiniState {velo_states, current_track_offset};

    const auto candidates = calculate_windows(
      shared_active_tracks[threadIdx.y],
      layer,
      velo_state,
      fudge_factors,
      ut_hits,
      ut_hit_offsets,
      dev_ut_dxDy,
      dev_unique_sector_xs,
      dev_unique_x_sector_layer_offsets,
      velo_tracks);

    // Write the windows in SoA style
    short* windows_layers = dev_windows_layers + event_tracks_offset * CompassUT::num_elems * UT::Constants::n_layers;

    const int track_pos = UT::Constants::n_layers * number_of_tracks_event;
    const int layer_pos = layer * number_of_tracks_event + shared_active_tracks[threadIdx.y];
    windows_layers[0 * track_pos + layer_pos] = std::get<0>(candidates) - layer_offset;            // first_candidate
    windows_layers[1 * track_pos + layer_pos] = std::get<2>(candidates) - layer_offset;            // left_group_first
    windows_layers[2 * track_pos + layer_pos] = std::get<4>(candidates) - layer_offset;            // right_group_first
    windows_layers[3 * track_pos + layer_pos] = std::get<6>(candidates) - layer_offset;            // left2_group_first
    windows_layers[4 * track_pos + layer_pos] = std::get<8>(candidates) - layer_offset;            // right2_group_first
    windows_layers[5 * track_pos + layer_pos] = std::get<1>(candidates) - std::get<0>(candidates); // last_size
    windows_layers[6 * track_pos + layer_pos] = std::get<3>(candidates) - std::get<2>(candidates); // left_size_last
    windows_layers[7 * track_pos + layer_pos] = std::get<5>(candidates) - std::get<4>(candidates); // right_size_first
    windows_layers[8 * track_pos + layer_pos] = std::get<7>(candidates) - std::get<6>(candidates); // left2_size_last
    windows_layers[9 * track_pos + layer_pos] = std::get<9>(candidates) - std::get<8>(candidates); // right2_size_first
  }
}
