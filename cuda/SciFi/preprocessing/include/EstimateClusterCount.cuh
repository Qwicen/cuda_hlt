#include "SciFiDefinitions.cuh"
__global__ void estimate_cluster_count(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  uint* scifi_hit_count,
  char* scifi_geometry
);
