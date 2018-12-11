#include "CalculateWindows.cuh"
#include "SearchWindows.cuh"
#include "Handler.cuh"
#include <tuple>

__global__ void ut_search_windows(
  uint* dev_ut_hits, // actual hit content
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage, // semi_prefixsum, offset to tracks
  uint* dev_velo_track_hit_number,
  uint* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  const uint* dev_unique_x_sector_layer_offsets, // prefixsum to point to the x hit of the sector, per layer
  const float* dev_unique_sector_xs,             // list of xs that define the groups
  int* dev_windows_layers)
{
  const uint number_of_events           = gridDim.x;
  const uint event_number               = blockIdx.x;
  const int layer                       = threadIdx.x;
  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint total_number_of_hits       = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks{
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states{dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset    = velo_tracks.tracks_offset(event_number);

  UTHitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  UTHits ut_hits {dev_ut_hits, total_number_of_hits};

  const float* fudge_factors = &(dev_ut_magnet_tool->dxLayTable[0]);

  for (int i = threadIdx.y; i < number_of_tracks_event; i += blockDim.y) {
    const uint current_track_offset = event_tracks_offset + i;

    const auto velo_state = MiniState{velo_states, current_track_offset};
    if (!velo_states.backward[current_track_offset]) {
      // Using Mini State with only x, y, tx, ty and z
      
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

        // Write the windows in SoA style
        // Write the index of candidate, then the size of the window (from, size, from, size....)
        int* windows_layers = dev_windows_layers + event_tracks_offset * NUM_ELEMS * VeloUTTracking::n_layers;
        windows_layers[(number_of_tracks_event * (0 + NUM_ELEMS * layer)) + i] = std::get<0>(candidates); // first_candidate
        windows_layers[(number_of_tracks_event * (1 + NUM_ELEMS * layer)) + i] = std::get<1>(candidates) - std::get<0>(candidates); // last_candidate
        windows_layers[(number_of_tracks_event * (2 + NUM_ELEMS * layer)) + i] = std::get<2>(candidates); // left_group_first
        windows_layers[(number_of_tracks_event * (3 + NUM_ELEMS * layer)) + i] = std::get<3>(candidates) - std::get<2>(candidates); // left_group_last
        windows_layers[(number_of_tracks_event * (4 + NUM_ELEMS * layer)) + i] = std::get<4>(candidates); // right_group_first
        windows_layers[(number_of_tracks_event * (5 + NUM_ELEMS * layer)) + i] = std::get<5>(candidates) - std::get<4>(candidates); // right_group_first
        windows_layers[(number_of_tracks_event * (6 + NUM_ELEMS * layer)) + i] = std::get<6>(candidates); // left2_group_first
        windows_layers[(number_of_tracks_event * (7 + NUM_ELEMS * layer)) + i] = std::get<7>(candidates) - std::get<6>(candidates); // left2_group_last
        windows_layers[(number_of_tracks_event * (8 + NUM_ELEMS * layer)) + i] = std::get<8>(candidates); // right2_group_first
        windows_layers[(number_of_tracks_event * (9 + NUM_ELEMS * layer)) + i] = std::get<9>(candidates) - std::get<8>(candidates); // right2_group_first

        // const int total_offset = NUM_ELEMS * VeloUTTracking::n_layers * current_track_offset + NUM_ELEMS * layer;
        // dev_windows_layers[total_offset]     = std::get<0>(candidates); // first_candidate
        // dev_windows_layers[total_offset + 1] = std::get<1>(candidates); // last_candidate
        // dev_windows_layers[total_offset + 2] = std::get<2>(candidates); // left_group_first
        // dev_windows_layers[total_offset + 3] = std::get<3>(candidates); // left_group_last
        // dev_windows_layers[total_offset + 4] = std::get<4>(candidates); // right_group_first
        // dev_windows_layers[total_offset + 5] = std::get<5>(candidates); // right_group_last
        // dev_windows_layers[total_offset + 6] = std::get<6>(candidates); // left2_group_first
        // dev_windows_layers[total_offset + 7] = std::get<7>(candidates); // left2_group_last
        // dev_windows_layers[total_offset + 8] = std::get<8>(candidates); // right2_group_first
        // dev_windows_layers[total_offset + 9] = std::get<9>(candidates); // right2_group_last
      }
    }
  }
}
