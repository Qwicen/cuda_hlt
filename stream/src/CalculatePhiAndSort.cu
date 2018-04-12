#include "../include/CalculatePhiAndSort.cuh"

void CalculatePhiAndSort::operator()() {
  calculatePhiAndSort<<<num_blocks, num_threads, 0, *stream>>>(
    dev_module_cluster_start,
    dev_module_cluster_num,
    dev_velo_cluster_container,
    dev_hit_permutation
  );
}
