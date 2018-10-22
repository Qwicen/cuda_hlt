#pragma once

#include "Handler.cuh"

__device__ void fill_candidates_impl(
  short* h0_candidates,
  short* h2_candidates,
  const uint* module_hitStarts,
  const uint* module_hitNums,
  const float* hit_Phis,
  const uint hit_offset
);

__global__ void fill_candidates(
  uint* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  short* dev_h0_candidates,
  short* dev_h2_candidates
);

ALGORITHM(fill_candidates, fill_candidates_t)
