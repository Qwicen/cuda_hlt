#include "../include/PrefixSum.cuh"

__global__ void prefix_sum_scan(uint* dev_main_array, uint* dev_auxiliary_array, const uint array_size)
{
  // Note: The first block is already correctly populated.
  //       Start on the second block.
  const uint element = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

  if (element < array_size) {
    const uint cluster_offset = dev_auxiliary_array[blockIdx.x + 1];
    dev_main_array[element] += cluster_offset;
  }
}
