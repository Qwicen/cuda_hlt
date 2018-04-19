#include "../include/EstimateInputSize.cuh"

void EstimateInputSize::operator()() {
  estimate_input_size<<<num_blocks, num_threads, 0, *stream>>>(
    dev_raw_input,
    dev_raw_input_offsets,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates
  );
}
