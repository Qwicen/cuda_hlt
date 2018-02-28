#include "Definitions.cuh"

__global__ void velo_fit(
  const char* dev_input,
  Track* dev_tracks,
  int* dev_atomics_storage,
  VeloState* dev_velo_states,
  int32_t* dev_hit_temp,
  unsigned int* dev_event_offsets,
  unsigned int* dev_hit_offsets
);
