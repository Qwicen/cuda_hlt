#include <iostream>
#include <functional>
#include <algorithm>

#include "IPCut.cuh"
#include <AssociateConsolidated.cuh>
#include <IPCutValues.cuh>

__global__ void ip_cut(
  char* dev_kalman_velo_states,
  int* dev_atomics_velo,
  uint* dev_velo_track_hit_number,
  char* dev_velo_pv_ip,
  bool* dev_accepted_velo_tracks)
{

  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks {
    (uint*) dev_atomics_velo, dev_velo_track_hit_number, event_number, number_of_events};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  Associate::Consolidated::Table velo_pv_ip {dev_velo_pv_ip, velo_tracks.total_number_of_tracks};
  // The track <-> PV association table for this event
  auto pv_table = velo_pv_ip.event_table(velo_tracks, event_number);

  bool* accepted = dev_accepted_velo_tracks + event_tracks_offset;

  for (int i = threadIdx.x; i < number_of_tracks_event; i += blockDim.x) {
    accepted[i] = (pv_table.value[i] > Cuts::IP::baseline);
  }
}
