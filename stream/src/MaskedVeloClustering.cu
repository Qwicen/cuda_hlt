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

void MaskedVeloClustering::print_output(
  const uint number_of_events,
  const int print_max_per_module
) {
  std::vector<uint> module_cluster_start (number_of_events * 52 + 1);
  std::vector<uint> module_cluster_num (number_of_events * 52);
  cudaCheck(cudaMemcpyAsync(module_cluster_start.data(), dev_module_cluster_start, module_cluster_start.size() * sizeof(uint), cudaMemcpyDeviceToHost, *stream));
  cudaCheck(cudaMemcpyAsync(module_cluster_num.data(), dev_module_cluster_num, module_cluster_num.size() * sizeof(uint), cudaMemcpyDeviceToHost, *stream));

  const auto estimated_number_of_clusters = module_cluster_start[module_cluster_start.size() - 1];
  std::vector<uint32_t> velo_cluster_container (6 * estimated_number_of_clusters);
  cudaCheck(cudaMemcpyAsync(velo_cluster_container.data(), dev_velo_cluster_container, velo_cluster_container.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, *stream));

  float* cluster_xs = (float*) &velo_cluster_container[0];
  float* cluster_ys = (float*) &velo_cluster_container[estimated_number_of_clusters];
  float* cluster_zs = (float*) &velo_cluster_container[2 * estimated_number_of_clusters];
  uint32_t* cluster_ids = (uint32_t*) &velo_cluster_container[3 * estimated_number_of_clusters];

  for (uint i=0; i<number_of_events; ++i) {
    std::cout << "Event " << i << std::endl;
    for (uint module=0; module<52; ++module) {
      std::cout << " Module " << module << ":";
      const auto mod_start = module_cluster_start[52*i + module];
      for (uint cluster=0; cluster<module_cluster_num[52*i + module]; ++cluster) {
        if (print_max_per_module != -1 && cluster >= print_max_per_module) break;

        const auto x = cluster_xs[mod_start + cluster];
        const auto y = cluster_ys[mod_start + cluster];
        const auto z = cluster_zs[mod_start + cluster];
        const auto id = cluster_ids[mod_start + cluster];

        std::cout << " {" << x << ", " << y << ", " << z << " (#" << id << ")}";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}
