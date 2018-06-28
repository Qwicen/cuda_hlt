#include "../include/HandleCopyAndPrefixSumSingleBlock.cuh"

void CopyAndPrefixSumSingleBlock::operator()() {
  copy_and_prefix_sum_single_block<<<num_blocks, num_threads, 0, *stream>>>(
    dev_total_sum,
    dev_input_array,
    dev_output_array,
    array_size
  );
}
