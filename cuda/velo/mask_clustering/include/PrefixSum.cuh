#pragma once

__global__ void prefix_sum(
  unsigned int* dev_estimated_input_size,
  unsigned int array_size
);
