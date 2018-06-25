#include "../include/PrefixSum.cuh"

__global__ void prefix_sum_scan(
  uint* dev_estimated_input_size,
  uint* dev_cluster_offset,
  const uint array_size
) {
  // Note: The first block is already correctly populated.
  //       Start on the second block.
  const uint element = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

  if (element < array_size) {
    const uint cluster_offset = dev_cluster_offset[blockIdx.x + 1];
    dev_estimated_input_size[element] += cluster_offset;
  }
}
