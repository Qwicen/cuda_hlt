#pragma once

#include <stdint.h>
#include "ClusteringDefinitions.cuh"

__global__ void estimate_input_size(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  uint* dev_estimated_input_size,
  uint* dev_module_cluster_num,
  uint* dev_module_candidate_num,
  uint32_t* dev_cluster_candidates,
  uint8_t* dev_velo_candidate_ks
);
