#pragma once

#include <cstdint>
#include <cassert>
#include "ClusteringDefinitions.cuh"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsVelo.cuh"

__global__ void estimate_input_size(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  uint* dev_estimated_input_size,
  uint* dev_module_cluster_num,
  uint* dev_module_candidate_num,
  uint32_t* dev_cluster_candidates,
  const uint* dev_event_list,
  uint* dev_event_order,
  uint8_t* dev_velo_candidate_ks);

ALGORITHM(
  estimate_input_size,
  velo_estimate_input_size_t,
  ARGUMENTS(
    dev_velo_raw_input,
    dev_velo_raw_input_offsets,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates,
    dev_event_list,
    dev_event_order))
