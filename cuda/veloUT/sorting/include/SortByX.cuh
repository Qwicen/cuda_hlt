#pragma once

#include "VeloUTDefinitions.cuh"
#include "Sorting.cuh"

__global__ void sort_by_x(
  uint32_t* dev_ut_hits,
  uint* dev_hit_permutations
 );
