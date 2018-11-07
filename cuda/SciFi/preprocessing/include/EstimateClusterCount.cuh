#pragma once

#include "SciFiDefinitions.cuh"
#include "Handler.cuh"

__global__ void estimate_cluster_count(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  uint* scifi_hit_count,
  char* scifi_geometry
);

ALGORITHM(estimate_cluster_count, estimate_cluster_count_t)
