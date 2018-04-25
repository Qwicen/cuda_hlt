#pragma once

#include "../../../cuda/velo/mask_clustering/include/PrefixSum.cuh"
#include "Handler.cuh"

struct PrefixSumSingleBlock : public Handler {
  // Call parameters
  uint* dev_total_sum;
  uint* dev_cluster_offset;
  uint array_size;

  PrefixSumSingleBlock() = default;

  void setParameters(
    uint* param_dev_total_sum,
    uint* param_dev_cluster_offset,
    uint param_array_size
  ) {
    dev_total_sum = param_dev_total_sum;
    dev_cluster_offset = param_dev_cluster_offset;
    array_size = param_array_size;
  }

  void operator()();
};
