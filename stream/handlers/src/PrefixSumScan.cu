#include "../include/PrefixSumScan.cuh"

void PrefixSumScan::operator()() {
  prefix_sum_scan<<<num_blocks, num_threads, 0, *stream>>>(
    dev_estimated_input_size,
    dev_cluster_offset,
    array_size
  );
}
