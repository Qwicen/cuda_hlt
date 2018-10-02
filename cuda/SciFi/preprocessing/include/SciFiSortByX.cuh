#pragma once
#include "SciFiDefinitions.cuh"
#include "Sorting.cuh"

__global__ void scifi_sort_by_x(
  uint* scifi_hits,
  uint32_t* scifi_hit_count,
  uint* scifi_hit_permutations
 );
