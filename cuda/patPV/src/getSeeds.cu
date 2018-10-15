#include "getSeeds.cuh"

 __global__ void getSeeds(
    VeloState* dev_velo_states,
  int * dev_atomics_storage,
  XYZPoint * dev_seeds,
  uint * dev_number_seed) {


  int event_number = blockIdx.x;
   //int * number_of_tracks = dev_atomics_storage;
   //int * acc_tracks = dev_atomics_storage + number_of_events;

  int number_of_tracks = dev_atomics_storage[event_number];

  XYZPoint point;
  point.x = 1.;
  point.y = 1.;
  point.z = 1.;
  int number_seeds = 0;
  for(int i = 0; i < number_of_tracks; i++) {
    point.x = i;
  point.y = i;
  point.z = i;
  number_seeds++;
    dev_seeds[i] = point;
  }

  dev_number_seed[event_number] = number_seeds;

 };