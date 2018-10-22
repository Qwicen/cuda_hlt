#pragma once

#include <cstdint>
#include <cassert>
#include "ClusteringDefinitions.cuh"
#include "Handler.cuh"

__global__ void masked_velo_clustering(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint* dev_event_candidate_num,
  uint* dev_cluster_candidates,
  uint32_t* dev_velo_cluster_container,
  char* dev_velo_geometry,
  uint8_t* dev_velo_sp_patterns,
  float* dev_velo_sp_fx,
  float* dev_velo_sp_fy
);

ALGORITHM(masked_velo_clustering, masked_velo_clustering_t)
