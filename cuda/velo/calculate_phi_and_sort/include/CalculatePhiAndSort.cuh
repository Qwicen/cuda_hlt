#pragma once

#include <stdint.h>
#include "../../common/include/Definitions.cuh"

__device__ void calculatePhi(
  const uint* module_hitStarts,
  const uint* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  float* hit_Phis,
  unsigned short* hit_permutations
);

__device__ void sortByPhi(
  const uint number_of_hits,
  float* hit_Xs,
  float* hit_Ys,
  float* hit_Zs,
  uint* hit_IDs,
  int32_t* hit_temp,
  unsigned short* hit_permutations
);

__global__ void calculatePhiAndSort(
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint32_t* dev_velo_cluster_container,
  unsigned short* dev_hit_permutations
);
