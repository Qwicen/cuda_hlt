#include "PrefixSum.cuh"

/**
 * @brief Up-Sweep
 */
__device__ void up_sweep_512(uint* data_block)
{
  uint starting_elem = 1;
  for (uint i = 2; i <= 512; i <<= 1) {
    for (uint j = 0; j < (511 + blockDim.x) / i; ++j) {
      const uint element = starting_elem + (j * blockDim.x + threadIdx.x) * i;
      if (element < 512) {
        data_block[element] += data_block[element - (i >> 1)];
      }
    }
    starting_elem += i;
    __syncthreads();
  }
}

/**
 * @brief Down-sweep
 */
__device__ void down_sweep_512(uint* data_block)
{
  for (uint i = 512; i >= 2; i >>= 1) {
    for (uint j = 0; j < (511 + blockDim.x) / i; ++j) {
      const auto element = 511 - (j * blockDim.x + threadIdx.x) * i;
      if (element < 512) {
        const auto other_element = element - (i >> 1);
        const auto value = data_block[other_element];
        data_block[other_element] = data_block[element];
        data_block[element] += value;
      }
    }
    __syncthreads();
  }
}

/**
 * @brief Prefix sum elements in dev_main_array using
 *        dev_auxiliary_array to store intermediate values.
 *
 * @details This algorithm is the first of three in our Blelloch
 *          scan implementation
 *          https://www.youtube.com/watch?v=mmYv3Haj6uc
 *
 *          dev_auxiliary_array should have a size of at least
 *          ceiling(size(dev_main_array) / 512).
 *
 *          Note: 512 is the block size, optimal for the maximum
 *          number of threads in a block, 1024 threads.
 */
__global__ void prefix_sum_reduce(uint* dev_main_array, uint* dev_auxiliary_array, const uint array_size)
{
  // Use a data block size of 512
  __shared__ uint data_block[512];

  // Let's do it in blocks of 512 (2^9)
  const uint last_block = array_size >> 9;
  if (blockIdx.x < last_block) {
    const uint first_elem = blockIdx.x << 9;

    // Load elements into shared memory, add prev_last_elem
    data_block[threadIdx.x] = dev_main_array[first_elem + threadIdx.x];
    data_block[threadIdx.x + blockDim.x] = dev_main_array[first_elem + threadIdx.x + blockDim.x];

    __syncthreads();

    up_sweep_512((uint*) &data_block[0]);

    if (threadIdx.x == 0) {
      dev_auxiliary_array[blockIdx.x] = data_block[511];
      data_block[511] = 0;
    }

    __syncthreads();

    down_sweep_512((uint*) &data_block[0]);

    // Store back elements
    // assert( first_elem + threadIdx.x + blockDim.x < number_of_events * VeloTracking::n_modules + 2);
    dev_main_array[first_elem + threadIdx.x] = data_block[threadIdx.x];
    dev_main_array[first_elem + threadIdx.x + blockDim.x] = data_block[threadIdx.x + blockDim.x];

    __syncthreads();
  }

  // Last block is special because
  // it may contain an unspecified number of elements
  else {
    const auto elements_remaining = array_size & 0x1FF; // % 512
    if (elements_remaining > 0) {
      const auto first_elem = array_size - elements_remaining;

      // Initialize all elements to zero
      data_block[threadIdx.x] = 0;
      data_block[threadIdx.x + blockDim.x] = 0;

      // Load elements
      const auto elem_index = first_elem + threadIdx.x;
      if (elem_index < array_size) {
        data_block[threadIdx.x] = dev_main_array[elem_index];
      }
      if ((elem_index + blockDim.x) < array_size) {
        data_block[threadIdx.x + blockDim.x] = dev_main_array[elem_index + blockDim.x];
      }

      __syncthreads();

      up_sweep_512((uint*) &data_block[0]);

      // Store sum of all elements
      if (threadIdx.x == 0) {
        dev_auxiliary_array[blockIdx.x] = data_block[511];
        data_block[511] = 0;
      }

      __syncthreads();

      down_sweep_512((uint*) &data_block[0]);

      // Store back elements
      if (elem_index < array_size) {
        dev_main_array[elem_index] = data_block[threadIdx.x];
      }
      if ((elem_index + blockDim.x) < array_size) {
        dev_main_array[elem_index + blockDim.x] = data_block[threadIdx.x + blockDim.x];
      }
    }
  }
}
