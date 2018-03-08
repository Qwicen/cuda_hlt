#include "Definitions.cuh"

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  Track* dev_tracks,
  const unsigned int number_of_events
);
