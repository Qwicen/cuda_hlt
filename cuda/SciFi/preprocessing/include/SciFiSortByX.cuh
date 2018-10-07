#pragma once

#include "SciFiDefinitions.cuh"

__global__ void scifi_sort_by_x(
  char* scifi_hits,
  uint32_t* scifi_hit_count,
  uint* scifi_hit_permutations
 );
