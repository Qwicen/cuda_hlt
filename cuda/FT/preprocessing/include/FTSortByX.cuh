#pragma once
#include "FTDefinitions.cuh"
#include "Sorting.cuh"

__global__ void ft_sort_by_x(
  char* dev_ft_hits,
  uint32_t* dev_ft_hit_count,
  uint* dev_ft_hit_permutations
 );
