#pragma once

#include "../../../cuda/velo/prefix_sum/include/PrefixSum.cuh"
#include "Handler.cuh"

struct PrefixSumReduce : public Handler {
  // Call parameters
  uint* dev_estimated_input_size;
  uint* dev_cluster_offset;
  uint array_size;

  PrefixSumReduce() = default;

  void setParameters(
    uint* param_dev_estimated_input_size,
    uint* param_dev_cluster_offset,
    uint param_array_size
  ) {
    dev_estimated_input_size = param_dev_estimated_input_size;
    dev_cluster_offset = param_dev_cluster_offset;
    array_size = param_array_size;
  }

  void operator()();
};
