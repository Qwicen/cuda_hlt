#include "VeloUT.cuh"

__global__ void veloUT(
  VeloUTTracking::HitsSoA* dev_ut_hits
  ) {

  int i_event = blockIdx.x;
  float first_x = dev_ut_hits[i_event].xAtYEq0(0);

  if ( i_event == 0 )
    printf("first x = %f \n", first_x);
}
