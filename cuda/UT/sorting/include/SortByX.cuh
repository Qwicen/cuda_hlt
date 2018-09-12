#pragma once

#include "VeloUTDefinitions.cuh"
#include "UTDefinitions.cuh"
#include "Sorting.cuh"

__global__ void sort_by_x(
  uint32_t* dev_ut_hits,
  uint32_t* dev_ut_hit_count,
  uint* dev_hit_permutations
 );
