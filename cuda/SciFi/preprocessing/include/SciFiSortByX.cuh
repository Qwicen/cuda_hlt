#pragma once

#include "SciFiDefinitions.cuh"
#include "Handler.cuh"

__global__ void scifi_sort_by_x(
  uint* scifi_hits,
  uint32_t* scifi_hit_count,
  uint* scifi_hit_permutations
 );

ALGORITHM(scifi_sort_by_x, scifi_sort_by_x_t)
