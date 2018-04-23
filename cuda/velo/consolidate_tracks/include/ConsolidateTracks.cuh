#pragma once

#include "../../common/include/VeloDefinitions.cuh"

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const Track* dev_tracks,
  Track* dev_output_tracks
);
