#include "SearchWindows.cuh"
#include "CalculateWindows.cuh"
#include <tuple>

__global__ void ut_search_windows(
  uint* dev_ut_hits, // actual hit content
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage, // semi_prefixsum, offset to tracks
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  const uint* dev_unique_x_sector_layer_offsets, // prefixsum to point to the x hit of the sector, per layer
  const float* dev_unique_sector_xs,             // list of xs that define the groups
  int* dev_windows_layers)
{
  const uint number_of_events           = gridDim.x;
  const uint event_number               = blockIdx.x;
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

  const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  // const float* bdlTable     = &(dev_ut_magnet_tool->bdlTable[0]);

  const int layer = threadIdx.y;
  for (int i = threadIdx.x; i < number_of_tracks_event; i += blockDim.x) {
    const uint velo_states_index = event_tracks_offset + i;
    int first_candidate = -1, last_candidate = -1;
    
    if (!velo_states.backward[velo_states_index]) {
      // Using Mini State with only x, y, tx, ty and z
      const auto velo_state = MiniState{velo_states, velo_states_index};
      if (velo_track_in_UTA_acceptance(velo_state)) {
        const auto candidates = calculate_windows(
          i,
          layer,
          velo_state,
          fudgeFactors,
          ut_hits,
          ut_hit_offsets,
          dev_ut_dxDy,
          dev_unique_sector_xs,
          dev_unique_x_sector_layer_offsets,
          velo_tracks);

        first_candidate = std::get<0>(candidates);
        last_candidate = std::get<1>(candidates);
      }
    }

    // Save first and last candidates in the correct position of dev_windows_layers
    dev_windows_layers[2 * VeloUTTracking::n_layers * velo_tracks.track_offset(i) + 2 * layer]     = first_candidate;
    dev_windows_layers[2 * VeloUTTracking::n_layers * velo_tracks.track_offset(i) + 2 * layer + 1] = last_candidate;
  }
}
