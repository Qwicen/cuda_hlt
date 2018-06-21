#pragma once

#include "../../../cuda/velo/prefix_sum/include/PrefixSum.cuh"
#include "Handler.cuh"

struct CopyAndPrefixSumSingleBlock : public Handler {
  // Call parameters
  uint* dev_total_sum;
  uint* dev_input_array;
  uint* dev_output_array;
  uint array_size;

  CopyAndPrefixSumSingleBlock() = default;

  void setParameters(
    uint* param_dev_total_sum,
    uint* param_dev_input_array,
    uint* param_dev_output_array,
    uint param_array_size
  ) {
    dev_total_sum = param_dev_total_sum;
    dev_input_array = param_dev_input_array;
    dev_output_array = param_dev_output_array;
    array_size = param_array_size;
  }

  void operator()();
};
