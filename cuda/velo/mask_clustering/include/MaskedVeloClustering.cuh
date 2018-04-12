#pragma once

#include <stdint.h>

__global__ void masked_velo_clustering(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint* dev_event_candidate_num,
  uint* dev_cluster_candidates,
  uint32_t* dev_velo_cluster_container,
  char* dev_velo_geometry,
  unsigned char* dev_sp_patterns,
  unsigned char* dev_sp_sizes,
  float* dev_sp_fx,
  float* dev_sp_fy
);
