#include "HandleMaskedVeloClustering.cuh"

void MaskedVeloClustering::operator()() {
  masked_velo_clustering<<<num_blocks, num_threads, 0, *stream>>>(
    dev_raw_input,
    dev_raw_input_offsets,
    dev_module_cluster_start,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates,
    dev_velo_cluster_container,
    dev_velo_geometry
  );
}

void MaskedVeloClustering::print_output(
  const uint number_of_events,
  const int print_max_per_module
) {
  std::vector<uint> module_cluster_start (number_of_events * N_MODULES + 1);
  std::vector<uint> module_cluster_num (number_of_events * N_MODULES);
  cudaCheck(cudaMemcpyAsync(module_cluster_start.data(), dev_module_cluster_start, module_cluster_start.size() * sizeof(uint), cudaMemcpyDeviceToHost, *stream));
  cudaCheck(cudaMemcpyAsync(module_cluster_num.data(), dev_module_cluster_num, module_cluster_num.size() * sizeof(uint), cudaMemcpyDeviceToHost, *stream));

  const auto estimated_number_of_clusters = module_cluster_start[module_cluster_start.size() - 1];
  std::vector<uint32_t> velo_cluster_container (6 * estimated_number_of_clusters);
  cudaCheck(cudaMemcpyAsync(velo_cluster_container.data(), dev_velo_cluster_container, velo_cluster_container.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, *stream));

  float* cluster_xs = (float*) &velo_cluster_container[0];
  float* cluster_ys = (float*) &velo_cluster_container[estimated_number_of_clusters];
  float* cluster_zs = (float*) &velo_cluster_container[2 * estimated_number_of_clusters];
  uint32_t* cluster_ids = (uint32_t*) &velo_cluster_container[3 * estimated_number_of_clusters];

  // Print number of found clusters per event
  for (uint i=0; i<number_of_events; ++i) {
    uint found_clusters = 0;
    for (uint module=0; module<N_MODULES; ++module) {
      found_clusters += module_cluster_num[N_MODULES*i + module];
    }
    std::cout << "Event " << i << ": " << found_clusters << " clusters" << std::endl;
  }

  // // Print all clusters
  // for (uint i=0; i<number_of_events; ++i) {
  //   std::cout << "Event " << i << std::endl;
  //   for (uint module=0; module<N_MODULES; ++module) {
  //     std::cout << " Module " << module << ":";
  //     const auto mod_start = module_cluster_start[N_MODULES*i + module];
  //     for (uint cluster=0; cluster<module_cluster_num[N_MODULES*i + module]; ++cluster) {
  //       if (print_max_per_module != -1 && cluster >= print_max_per_module) break;

  //       const auto x = cluster_xs[mod_start + cluster];
  //       const auto y = cluster_ys[mod_start + cluster];
  //       const auto z = cluster_zs[mod_start + cluster];
  //       const auto id = cluster_ids[mod_start + cluster];

  //       std::cout << " {" << x << ", " << y << ", " << z << " (#" << id << ")}";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }

  // // Print checksum for every event
  // for (uint i=0; i<number_of_events; ++i) {
  //   float sum = 0;
  //   for (uint module=0; module<N_MODULES; ++module) {
  //     const auto mod_start = module_cluster_start[N_MODULES*i + module];
  //     for (uint cluster=0; cluster<module_cluster_num[N_MODULES*i + module]; ++cluster) {
  //       if (print_max_per_module != -1 && cluster >= print_max_per_module) break;
  //       const auto x = cluster_xs[mod_start + cluster];
  //       const auto y = cluster_ys[mod_start + cluster];
  //       const auto z = cluster_zs[mod_start + cluster];
  //       sum += x + y + z;
  //     }
  //   }
  //   std::cout << "Event " << i << ": " << sum << std::endl;
  // }
}

void MaskedVeloClustering::check(
  const char* host_events_pinned,
  const uint* host_event_offsets_pinned,
  const size_t host_events_pinned_size,
  const size_t host_event_offsets_pinned_size,
  const std::vector<char>& geometry,
  const uint number_of_events
) {
  std::vector<char> events (host_events_pinned_size);
  std::vector<uint> event_offsets (host_event_offsets_pinned_size);
  std::copy_n(host_events_pinned, host_events_pinned_size, events.data());
  std::copy_n(host_event_offsets_pinned, host_event_offsets_pinned_size, event_offsets.data());

  std::vector<uint> module_cluster_start (number_of_events * N_MODULES + 1);
  std::vector<uint> module_cluster_num (number_of_events * N_MODULES);
  cudaCheck(cudaMemcpyAsync(module_cluster_start.data(), dev_module_cluster_start, module_cluster_start.size() * sizeof(uint), cudaMemcpyDeviceToHost, *stream));
  cudaCheck(cudaMemcpyAsync(module_cluster_num.data(), dev_module_cluster_num, module_cluster_num.size() * sizeof(uint), cudaMemcpyDeviceToHost, *stream));

  const auto estimated_number_of_clusters = module_cluster_start[module_cluster_start.size() - 1];
  std::vector<uint32_t> velo_cluster_container (number_of_events * 2000 * 6);
  cudaCheck(cudaMemcpyAsync(velo_cluster_container.data(), dev_velo_cluster_container, velo_cluster_container.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, *stream));
  uint32_t* cluster_ids = (uint32_t*) &velo_cluster_container[3 * estimated_number_of_clusters];

  std::vector<std::vector<uint32_t>> found_lhcb_ids;
  for (uint i=0; i<number_of_events; ++i) {
    std::vector<uint32_t> event_found_lhcb_ids;
    for (uint module=0; module<N_MODULES; ++module) {
      const uint mod_num = module_cluster_num[N_MODULES*i + module];
      const uint mod_start = module_cluster_start[N_MODULES*i + module];
      for (uint cluster=0; cluster<mod_num; ++cluster) {
        const auto id = cluster_ids[mod_start + cluster];
        event_found_lhcb_ids.emplace_back(id);
      }
    }
    found_lhcb_ids.emplace_back(event_found_lhcb_ids);
  }

  float reconstruction_efficiency;
  float clone_fraction;
  float ghost_fraction;

  std::cout << "Clustering efficiency across " << number_of_events << " events:" << std::endl;

  checkClustering(
    geometry,
    events,
    event_offsets,
    found_lhcb_ids,
    reconstruction_efficiency,
    clone_fraction,
    ghost_fraction,
    false,
    2.0f
  );

  std::cout << "With LHCb ID comparison max distance 2.0:" << std::endl
    << " Reconstruction efficiency: " << reconstruction_efficiency << std::endl
    << " Clone fraction: " << clone_fraction << std::endl
    << " Ghost fraction: " << ghost_fraction << std::endl << std::endl;
}
