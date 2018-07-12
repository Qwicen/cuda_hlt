#pragma once

#include "../../common/include/VeloDefinitions.cuh"

__device__ void fill_candidates_impl(
  short* h0_candidates,
  short* h2_candidates,
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums,
  const float* hit_Phis,
  const uint hit_offset
);

__global__ void fill_candidates(
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  short* dev_h0_candidates,
  short* dev_h2_candidates
);
