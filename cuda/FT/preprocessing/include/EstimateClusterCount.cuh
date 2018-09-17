#include "FTDefinitions.cuh"
__global__ void estimate_cluster_count(
  char* dev_ft_raw_input,
  uint* dev_ft_raw_input_offsets,
  uint* dev_ft_hit_count,
  char* dev_ft_geometry
);
