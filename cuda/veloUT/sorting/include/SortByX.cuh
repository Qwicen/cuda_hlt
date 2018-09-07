#pragma once

#include "VeloUTDefinitions.cuh"
#include "Sorting.cuh"

__global__ void sort_by_x(
  VeloUTTracking::HitsSoA* dev_ut_hits,
  VeloUTTracking::HitsSoA* dev_ut_hits_sorted,
  uint* dev_hit_permutations
 );
