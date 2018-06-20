#include "../include/HandlePrefixSumReduce.cuh"

void PrefixSumReduce::operator()() {
  prefix_sum_reduce<<<num_blocks, num_threads, 0, *stream>>>(
    dev_estimated_input_size,
    dev_cluster_offset,
    array_size
  );
}
