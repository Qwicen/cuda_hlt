#include "HandleCalculatePhiAndSort.cuh"

void CalculatePhiAndSort::operator()() {
  calculatePhiAndSort<<<num_blocks, num_threads, 0, *stream>>>(
    dev_module_cluster_start,
    dev_module_cluster_num,
    dev_velo_cluster_container,
    dev_hit_permutation
  );
}

void CalculatePhiAndSort::print_output(
  const uint number_of_events,
  const int print_max_per_module
) {
  std::vector<uint> module_cluster_start (number_of_events * VeloTracking::n_modules + 1);
  std::vector<uint> module_cluster_num (number_of_events * VeloTracking::n_modules);
  cudaCheck(cudaMemcpyAsync(module_cluster_start.data(), dev_module_cluster_start, module_cluster_start.size() * sizeof(uint), cudaMemcpyDeviceToHost, *stream));
  cudaCheck(cudaMemcpyAsync(module_cluster_num.data(), dev_module_cluster_num, module_cluster_num.size() * sizeof(uint), cudaMemcpyDeviceToHost, *stream));

  const auto estimated_number_of_clusters = module_cluster_start[module_cluster_start.size() - 1];
  std::vector<uint32_t> velo_cluster_container (6 * estimated_number_of_clusters);
  cudaCheck(cudaMemcpyAsync(velo_cluster_container.data(), dev_velo_cluster_container, velo_cluster_container.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, *stream));

  float* cluster_ys = (float*) &velo_cluster_container[0];
  float* cluster_zs = (float*) &velo_cluster_container[estimated_number_of_clusters];
  uint32_t* cluster_ids = (uint32_t*) &velo_cluster_container[2 * estimated_number_of_clusters];
  float* cluster_phis = (float*) &velo_cluster_container[4 * estimated_number_of_clusters];
  float* cluster_xs = (float*) &velo_cluster_container[5 * estimated_number_of_clusters];

  for (uint i=0; i<number_of_events; ++i) {
    std::cout << "Event " << i << std::endl;
    for (uint module=0; module<VeloTracking::n_modules; ++module) {
      std::cout << " Module " << module << ":";
      const auto mod_start = module_cluster_start[VeloTracking::n_modules*i + module];
      for (uint cluster=0; cluster<module_cluster_num[VeloTracking::n_modules*i + module]; ++cluster) {
        if (print_max_per_module != -1 && cluster >= print_max_per_module) break;

        const auto x = cluster_xs[mod_start + cluster];
        const auto y = cluster_ys[mod_start + cluster];
        const auto z = cluster_zs[mod_start + cluster];
        const auto phi = cluster_phis[mod_start + cluster];
        const auto id = cluster_ids[mod_start + cluster];

        std::cout << " {" << x << ", " << y << ", " << z << ", " << phi << " (#" << id << ")}";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // for (uint i=0; i<number_of_events; ++i) {
  //   float phi_sum = 0.f;
  //   for (uint module=0; module<VeloTracking::n_modules; ++module) {
  //     const auto mod_start = module_cluster_start[VeloTracking::n_modules*i + module];
  //     for (uint cluster=0; cluster<module_cluster_num[VeloTracking::n_modules*i + module]; ++cluster) {
  //       if (print_max_per_module != -1 && cluster >= print_max_per_module) break;
  //       phi_sum += cluster_phis[mod_start + cluster];
  //     }
  //   }
  //   std::cout << "Event " << i << ": " << phi_sum << std::endl;
  // }
}
