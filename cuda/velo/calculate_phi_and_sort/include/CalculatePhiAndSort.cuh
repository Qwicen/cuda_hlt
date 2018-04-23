#pragma once

#include <stdint.h>
#include "../../common/include/VeloDefinitions.cuh"

__device__ void calculatePhi(
  const uint* module_hitStarts,
  const uint* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  float* hit_Phis,
  uint* hit_permutations
);

__device__ void sortByPhi(
  const uint event_hit_start,
  const uint event_number_of_hits,
  float* hit_Xs,
  float* hit_Ys,
  float* hit_Zs,
  uint* hit_IDs,
  int32_t* hit_temp,
  uint* hit_permutations
);

__global__ void calculatePhiAndSort(
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint32_t* dev_velo_cluster_container,
  uint* dev_hit_permutations
);
