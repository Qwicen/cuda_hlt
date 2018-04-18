#include "../include/PrefixSum.cuh"

void PrefixSum::operator()() {
  prefix_sum<<<num_blocks, num_threads, 0, *stream>>>(
    dev_estimated_input_size,
    array_size
  );
}
