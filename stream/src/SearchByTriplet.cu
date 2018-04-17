#include "../include/SearchByTriplet.cuh"

void SearchByTriplet::operator()() {
  searchByTriplet<<<num_blocks, num_threads, 0, *stream>>>(
    dev_velo_cluster_container,
    dev_module_cluster_start,
    dev_module_cluster_num,
    dev_tracks,
    dev_tracklets,
    dev_tracks_to_follow,
    dev_weak_tracks,
    dev_hit_used,
    dev_atomics_storage,
    dev_h0_candidates,
    dev_h2_candidates,
    dev_rel_indices
  );
}

void SearchByTriplet::print_output(
  const uint number_of_events
) {
  // Fetch clusters
  std::vector<uint> module_cluster_start (number_of_events * 52 + 1);
  std::vector<uint> module_cluster_num (number_of_events * 52);
  cudaCheck(cudaMemcpy(module_cluster_start.data(), dev_module_cluster_start, module_cluster_start.size() * sizeof(uint), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(module_cluster_num.data(), dev_module_cluster_num, module_cluster_num.size() * sizeof(uint), cudaMemcpyDeviceToHost));

  std::vector<short> h0_candidates (2 * 2000 * number_of_events);
  std::vector<short> h2_candidates (2 * 2000 * number_of_events);
  cudaCheck(cudaMemcpy(h0_candidates.data(), dev_h0_candidates, h0_candidates.size() * sizeof(short), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h2_candidates.data(), dev_h2_candidates, h2_candidates.size() * sizeof(short), cudaMemcpyDeviceToHost));

  // Print h0 candidates, h2 candidates
  for (int i=0; i<number_of_events; ++i) {
    int sum_h0 = 0;
    int sum_h2 = 0;
    for (int module=0; module<52; ++module) {
      const auto h_start = module_cluster_start[52*i + module];
      for (int n=0; n<module_cluster_num[52*i + module]; ++n) {
        sum_h0 += h0_candidates[2*(h_start + n)];
        sum_h0 += h0_candidates[2*(h_start + n) + 1];
        sum_h2 += h2_candidates[2*(h_start + n)];
        sum_h2 += h2_candidates[2*(h_start + n) + 1];
      }
    }
    std::cout << "Event " << i << ": " << sum_h0 << ", " << sum_h2 << std::endl;
  }

  // const auto estimated_number_of_clusters = module_cluster_start[module_cluster_start.size() - 1];
  // std::vector<uint32_t> velo_cluster_container (6 * estimated_number_of_clusters);
  // cudaCheck(cudaMemcpy(velo_cluster_container.data(), dev_velo_cluster_container, velo_cluster_container.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // float* cluster_ys = (float*) &velo_cluster_container[0];
  // float* cluster_zs = (float*) &velo_cluster_container[estimated_number_of_clusters];
  // uint32_t* cluster_ids = (uint32_t*) &velo_cluster_container[2 * estimated_number_of_clusters];
  // float* cluster_phis = (float*) &velo_cluster_container[4 * estimated_number_of_clusters];
  // float* cluster_xs = (float*) &velo_cluster_container[5 * estimated_number_of_clusters];

  // // Fetch tracks
  // std::vector<int> number_of_tracks (number_of_events);
  // std::vector<Track> tracks (number_of_events * MAX_TRACKS);
  // cudaCheck(cudaMemcpy(number_of_tracks.data(), dev_atomics_storage, number_of_tracks.size() * sizeof(int), cudaMemcpyDeviceToHost));
  // cudaCheck(cudaMemcpy(tracks.data(), dev_tracks, tracks.size() * sizeof(Track), cudaMemcpyDeviceToHost));

  // // Print
  // for (int i=0; i<number_of_events; ++i) {
  //   std::cout << "Event " << i << ": Found " << number_of_tracks[i] << " tracks" << std::endl;
  //   for (int j=0; j<number_of_tracks[i]; ++j) {
  //     const auto track = tracks[i*MAX_TRACKS + j];
  //     std::cout << " Track #" << j << ", " << track.hitsNum << " hits:";
  //     for (int k=0; k<track.hitsNum; ++k) {
  //       const auto x = cluster_xs[track.hits[k]];
  //       const auto y = cluster_ys[track.hits[k]];
  //       const auto z = cluster_zs[track.hits[k]];
  //       const auto id = cluster_ids[track.hits[k]];

  //       std::cout << " {#" << track.hits[k] << ", " << x << ", " << y << ", " << z << ", (#" << id << ")}";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
}
