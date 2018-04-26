#pragma once

#include "../../../cuda/velo/mask_clustering/include/EstimateInputSize.cuh"
#include "Handler.cuh"

struct EstimateInputSize : public Handler {
  // Call parameters
  char* dev_raw_input;
  uint* dev_raw_input_offsets;
  uint* dev_estimated_input_size;
  uint* dev_module_cluster_num;
  uint* dev_module_candidate_num;
  uint32_t* dev_cluster_candidates;

  EstimateInputSize() = default;

  void setParameters(
    char* param_dev_raw_input,
    uint* param_dev_raw_input_offsets,
    uint* param_dev_estimated_input_size,
    uint* param_dev_module_cluster_num,
    uint* param_dev_module_candidate_num,
    uint32_t* param_dev_cluster_candidates
  ) {
    dev_raw_input = param_dev_raw_input;
    dev_raw_input_offsets = param_dev_raw_input_offsets;
    dev_estimated_input_size = param_dev_estimated_input_size;
    dev_module_cluster_num = param_dev_module_cluster_num;
    dev_module_candidate_num = param_dev_module_candidate_num;
    dev_cluster_candidates = param_dev_cluster_candidates;
  }

  void operator()();
};
