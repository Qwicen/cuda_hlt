#pragma once

#include <stdint.h>
#include "VeloDefinitions.cuh"

__global__ void velo_fit(
  const uint32_t* dev_velo_cluster_container,
  const uint* dev_module_cluster_start,
  const int* dev_atomics_storage,
  const Track* dev_tracks,
  VeloState* dev_velo_states,
  const bool is_consolidated = false
);
