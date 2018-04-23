#include "PrefixSum.cuh"

/**
 * @brief Up-Sweep (reduce)
 */
__device__ void up_sweep(
  uint* data_block
) {
  uint starting_elem = 1;
  for (uint i=2; i<=2048; i<<=1) {
    for (uint j=0; j<(2047 + blockDim.x) / i; ++j) {
      const uint element = starting_elem + (j*blockDim.x + threadIdx.x) * i;
      if (element < 2048) {
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
__device__ void down_sweep(
  uint* data_block
) {
  for (uint i=2048; i>=2; i>>=1) {
    for (uint j=0; j<(2047 + blockDim.x) / i; ++j) {
      const auto element = 2047 - (j*blockDim.x + threadIdx.x) * i;
      if (element < 2048) {
        const auto other_element = element - (i>>1);
        const auto value = data_block[other_element];
        data_block[other_element] = data_block[element];
        data_block[element] += value;
      }
    }
    __syncthreads();
  }
}

__global__ void prefix_sum(
  uint* dev_estimated_input_size,
  const uint array_size
) {
  // Prefix sum of elements in dev_estimated_input_size
  // Using Blelloch scan https://www.youtube.com/watch?v=mmYv3Haj6uc
  __shared__ uint data_block [2048];

  // Let's do it in blocks of 2048 (2^11)
  unsigned prev_last_elem = 0;
  for (uint block=0; block<(array_size>>11); ++block) {
    const uint first_elem = block << 11;

    // Load elements into shared memory, add prev_last_elem
    data_block[2*threadIdx.x] = dev_estimated_input_size[first_elem + 2*threadIdx.x];
    data_block[2*threadIdx.x + 1] = dev_estimated_input_size[first_elem + 2*threadIdx.x + 1];

    __syncthreads();

    up_sweep((uint*) &data_block[0]);

    const uint new_last_elem = data_block[2047];

    __syncthreads();
    data_block[2047] = 0;
    __syncthreads();

    down_sweep((uint*) &data_block[0]);

    // Store back elements
    dev_estimated_input_size[first_elem + 2*threadIdx.x] = data_block[2*threadIdx.x] + prev_last_elem;
    dev_estimated_input_size[first_elem + 2*threadIdx.x + 1] = data_block[2*threadIdx.x + 1] + prev_last_elem;
    prev_last_elem += new_last_elem;

    __syncthreads();
  }

  // Last iteration is special because
  // it may contain an unspecified number of elements
  const auto elements_remaining = array_size & 0x7FF; // % 2048
  if (elements_remaining > 0) {
    const auto first_elem = array_size - elements_remaining;

    // Initialize all elements to zero
    data_block[2*threadIdx.x] = 0;
    data_block[2*threadIdx.x + 1] = 0;

    // Load elements
    const auto elem_index = first_elem + 2 * threadIdx.x;
    if (elem_index < array_size) {
      data_block[2*threadIdx.x] = dev_estimated_input_size[elem_index];
    }
    if ((elem_index+1) < array_size) {
      data_block[2*threadIdx.x + 1] = dev_estimated_input_size[elem_index + 1];
    }

    __syncthreads();

    up_sweep((uint*) &data_block[0]);

    // Store sum of all elements
    if (threadIdx.x==0) {
      dev_estimated_input_size[array_size] = prev_last_elem + data_block[2047];
    }

    __syncthreads();
    data_block[2047] = 0;
    __syncthreads();

    down_sweep((uint*) &data_block[0]);

    // Store back elements
    if (elem_index < array_size) {
      dev_estimated_input_size[elem_index] = data_block[2*threadIdx.x] + prev_last_elem;
    }
    if ((elem_index+1) < array_size) {
      dev_estimated_input_size[elem_index + 1] = data_block[2*threadIdx.x + 1] + prev_last_elem;
    }
  }
}
