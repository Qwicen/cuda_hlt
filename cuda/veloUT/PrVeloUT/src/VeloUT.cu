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

  int i_event = blockIdx.x;
  float first_x = dev_ut_hits[i_event].xAtYEq0(0);

  if ( i_event == 0 )
    printf("first x = %f \n", first_x);
}
