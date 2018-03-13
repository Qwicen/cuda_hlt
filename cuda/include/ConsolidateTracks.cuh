#include "Definitions.cuh"

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const Track* dev_tracks,
  Track* dev_output_tracks,
  unsigned int* dev_hit_offsets,
  unsigned short* dev_hit_permutation
);
