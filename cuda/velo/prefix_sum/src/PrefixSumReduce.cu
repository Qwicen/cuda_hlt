#include "PrefixSum.cuh"

/**
 * @brief Up-Sweep
 */
__device__ void up_sweep_512(
  uint* data_block
) {
  uint starting_elem = 1;
  for (uint i=2; i<=512; i<<=1) {
    for (uint j=0; j<(511 + blockDim.x) / i; ++j) {
      const uint element = starting_elem + (j*blockDim.x + threadIdx.x) * i;
      if (element < 512) {
        data_block[element] += data_block[element - (i>>1)];
      }
    }
    starting_elem += i;
    __syncthreads();
  }
}

/**
 * @brief Down-sweep
 */
__device__ void down_sweep_512(
  uint* data_block
) {
  for (uint i=512; i>=2; i>>=1) {
    for (uint j=0; j<(511 + blockDim.x) / i; ++j) {
      const auto element = 511 - (j*blockDim.x + threadIdx.x) * i;
      if (element < 512) {
        const auto other_element = element - (i>>1);
        const auto value = data_block[other_element];
        data_block[other_element] = data_block[element];
        data_block[element] += value;
      }
    }
    __syncthreads();
  }
}

__global__ void prefix_sum_reduce(
  uint* dev_estimated_input_size,
  uint* dev_cluster_offset,
  const uint array_size // VeloTracking::n_modules * number_of_events
) {
  
  // Prefix sum of elements in dev_estimated_input_size
  // Using Blelloch scan https://www.youtube.com/watch?v=mmYv3Haj6uc
  __shared__ uint data_block [512];

  // Let's do it in blocks of 512 (2^9)
  const uint last_block = array_size >> 9;
  if (blockIdx.x < last_block) {
    const uint first_elem = blockIdx.x << 9;

    // Load elements into shared memory, add prev_last_elem
    data_block[threadIdx.x] = dev_estimated_input_size[first_elem + threadIdx.x];
    data_block[threadIdx.x + blockDim.x] = dev_estimated_input_size[first_elem + threadIdx.x + blockDim.x];

    __syncthreads();

    up_sweep_512((uint*) &data_block[0]);

    if (threadIdx.x == 0) {
      dev_cluster_offset[blockIdx.x] = data_block[511];
      data_block[511] = 0;
    }

    __syncthreads();

    down_sweep_512((uint*) &data_block[0]);

    // Store back elements
    //assert( first_elem + threadIdx.x + blockDim.x < number_of_events * VeloTracking::n_modules + 2);
    dev_estimated_input_size[first_elem + threadIdx.x] = data_block[threadIdx.x];
    dev_estimated_input_size[first_elem + threadIdx.x + blockDim.x] = data_block[threadIdx.x + blockDim.x];

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
        data_block[threadIdx.x] = dev_estimated_input_size[elem_index];
      }
      if ((elem_index+blockDim.x) < array_size) {
        data_block[threadIdx.x + blockDim.x] = dev_estimated_input_size[elem_index + blockDim.x];
      }

      __syncthreads();

      up_sweep_512((uint*) &data_block[0]);

      // Store sum of all elements
      if (threadIdx.x == 0) {
        dev_cluster_offset[blockIdx.x] = data_block[511];
        data_block[511] = 0;
      }
      
      __syncthreads();

      down_sweep_512((uint*) &data_block[0]);

      // Store back elements
      if (elem_index < array_size) {
        dev_estimated_input_size[elem_index] = data_block[threadIdx.x];
      }
      if ((elem_index+blockDim.x) < array_size) {
        dev_estimated_input_size[elem_index + blockDim.x] = data_block[threadIdx.x + blockDim.x];
      }
    }
  }
}
