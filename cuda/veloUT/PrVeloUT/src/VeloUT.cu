#include "VeloUT.cuh"

__global__ void veloUT(
  VeloUTTracking::HitsSoA* dev_ut_hits,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states,
  VeloUTTracking::TrackUT* dev_veloUT_tracks,
  PrUTMagnetTool* dev_ut_magnet_tool
) {

  const int number_of_events = gridDim.x;
  const int event_number = blockIdx.x;
  
  VeloUTTracking::HitsSoA* dev_ut_hits_event = dev_ut_hits + event_number;
  const int number_of_tracks = *(dev_atomics_storage + event_number);
  const int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  const int accumulated_tracks = accumulated_tracks_base_pointer[event_number];
  
  
  
}
