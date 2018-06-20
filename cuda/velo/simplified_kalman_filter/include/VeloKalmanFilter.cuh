#pragma once

#include <stdint.h>
#include "../../common/include/VeloDefinitions.cuh"


__global__ void velo_fit(
  const uint32_t* dev_velo_cluster_container,
  const uint* dev_module_cluster_start,
  const int* dev_atomics_storage,
  const TrackHits* dev_tracks,
  VeloState* dev_velo_states
);


