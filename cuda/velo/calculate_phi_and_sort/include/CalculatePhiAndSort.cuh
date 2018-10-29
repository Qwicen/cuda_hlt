#pragma once

#include <cstdint>
#include <cassert>
#include "VeloDefinitions.cuh"
#include "Handler.cuh"

__device__ void calculate_phi(
  const uint* module_hitStarts,
  const uint* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  float* hit_Phis,
  uint* hit_permutations
);

__device__ void sort_by_phi(
  const uint event_hit_start,
  const uint event_number_of_hits,
  float* hit_Xs,
  float* hit_Ys,
  float* hit_Zs,
  uint* hit_IDs,
  int32_t* hit_temp,
  uint* hit_permutations
);

__global__ void calculate_phi_and_sort(
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint32_t* dev_velo_cluster_container,
  uint* dev_hit_permutations
);

ALGORITHM(calculate_phi_and_sort, calculate_phi_and_sort_t)
