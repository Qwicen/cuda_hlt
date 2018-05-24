#include "../include/HandlePrefixSumSingleBlock.cuh"

void PrefixSumSingleBlock::operator()() {
  prefix_sum_single_block<<<num_blocks, num_threads, 0, *stream>>>(
    dev_total_sum,
    dev_cluster_offset,
    array_size
  );
}
