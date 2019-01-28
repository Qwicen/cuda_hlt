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
  int* dev_windows_layers,
  bool* dev_accepted_velo_tracks)
{
  const uint number_of_events           = gridDim.x;
  const uint event_number               = blockIdx.x;
  const int layer                       = threadIdx.y;
  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint total_number_of_hits       = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks{
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states{dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset    = velo_tracks.tracks_offset(event_number);

  UT::HitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  UT::Hits ut_hits {dev_ut_hits, total_number_of_hits};

  const float* fudge_factors = &(dev_ut_magnet_tool->dxLayTable[0]);

  for (int i = threadIdx.x; i < number_of_tracks_event; i += blockDim.x) {
    const uint current_track_offset = event_tracks_offset + i;
    int first_candidate = -1, last_candidate = -1;
    int left_group_first_candidate = -1, left_group_last_candidate = -1;
    int right_group_first_candidate = -1, right_group_last_candidate = -1;

    if (!velo_states.backward[current_track_offset] && dev_accepted_velo_tracks[current_track_offset]) {
      // Using Mini State with only x, y, tx, ty and z
      const auto velo_state = MiniState{velo_states, current_track_offset};
      if (velo_track_in_UTA_acceptance(velo_state)) {
        const auto candidates = calculate_windows(
          i,
          layer,
          velo_state,
          fudge_factors,
          ut_hits,
          ut_hit_offsets,
          dev_ut_dxDy,
          dev_unique_sector_xs,
          dev_unique_x_sector_layer_offsets,
          velo_tracks);

        first_candidate = std::get<0>(candidates);
        last_candidate  = std::get<1>(candidates);
        left_group_first_candidate = std::get<2>(candidates);
        left_group_last_candidate = std::get<3>(candidates);
        right_group_first_candidate = std::get<4>(candidates);
        right_group_last_candidate = std::get<5>(candidates);
      }
    }
    // Save first and last candidates in the correct position of dev_windows_layers
    dev_windows_layers[6 * UT::Constants::n_layers * current_track_offset + 6 * layer]     = first_candidate;
    dev_windows_layers[6 * UT::Constants::n_layers * current_track_offset + 6 * layer + 1] = last_candidate;
    dev_windows_layers[6 * UT::Constants::n_layers * current_track_offset + 6 * layer + 2] = left_group_first_candidate;
    dev_windows_layers[6 * UT::Constants::n_layers * current_track_offset + 6 * layer + 3] = left_group_last_candidate;
    dev_windows_layers[6 * UT::Constants::n_layers * current_track_offset + 6 * layer + 4] = right_group_first_candidate;
    dev_windows_layers[6 * UT::Constants::n_layers * current_track_offset + 6 * layer + 5] = right_group_last_candidate;
  }
}
