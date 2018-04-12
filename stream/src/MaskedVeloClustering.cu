#include "../include/MaskedVeloClustering.cuh"

void MaskedVeloClustering::operator()() {
  masked_velo_clustering<<<num_blocks, num_threads, 0, *stream>>>(
    dev_raw_input,
    dev_raw_input_offsets,
    dev_module_cluster_start,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates,
    dev_velo_cluster_container,
    dev_velo_geometry,
    dev_sp_patterns,
    dev_sp_sizes,
    dev_sp_fx,
    dev_sp_fy
  );
}
