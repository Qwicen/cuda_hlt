#include "../include/Stream.cuh"

cudaError_t Stream::initialize(
  const std::vector<char>& raw_events,
  const std::vector<uint>& event_offsets,
  const std::vector<char>& geometry,
  const uint number_of_events,
  const size_t param_starting_events_size,
  const bool param_transmit_host_to_device,
  const bool param_transmit_device_to_host,
  const bool param_do_consolidate,
  const bool param_do_simplified_kalman_filter,
  const uint param_stream_number
) {
  cudaCheck(cudaStreamCreate(&stream));
  cudaCheck(cudaEventCreate(&cuda_generic_event));
  stream_number = param_stream_number;
  transmit_host_to_device = param_transmit_host_to_device;
  transmit_device_to_host = param_transmit_device_to_host;
  do_consolidate = param_do_consolidate;
  do_simplified_kalman_filter = param_do_simplified_kalman_filter;

  // Blocks and threads for each algorithm
  num_blocks = dim3(number_of_events);
  estimate_input_size_blocks = dim3(number_of_events);
  prefix_sum_blocks = dim3(1);
  masked_velo_clustering_blocks = dim3(number_of_events);
  consolidate_blocks = dim3(number_of_events);

  estimate_input_size_threads = dim3(4, 208);
  prefix_sum_threads = dim3(1024);
  masked_velo_clustering_threads = dim3(256);
  sort_num_threads = dim3(64);
  sbt_num_threads = dim3(NUMTHREADS_X);
  consolidate_num_threads = dim3(32);
  simplified_kalman_num_threads = dim3(1024);

  // velo cluster container contains:
  // - cluster_xs
  // - cluster_ys
  // - cluster_zs
  // - cluster_ids
  // - cluster_phis
  // - temporary
  // 
  // The temporary is required to do the sorting in an efficient manner
  velo_cluster_container_size = number_of_events * 2000 * 6;

  // Data preparation
  // Populate velo geometry
  cudaCheck(cudaMalloc((void**)&dev_velo_geometry, geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_velo_geometry, geometry.data(), geometry.size(), cudaMemcpyHostToDevice, stream));
  
  // Allocate buffers for algorithms
  // Clustering
  cudaCheck(cudaMalloc((void**)&dev_raw_input, param_starting_events_size));
  cudaCheck(cudaMalloc((void**)&dev_raw_input_offsets, event_offsets.size() * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_estimated_input_size, (number_of_events * 52 + 2) * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_module_cluster_num, number_of_events * 52 * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_module_candidate_num, number_of_events * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_cluster_candidates, number_of_events * 2000 * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_velo_cluster_container, velo_cluster_container_size * sizeof(uint)));

  // phi and sort
  cudaCheck(cudaMalloc((void**)&dev_hit_permutation, average_number_of_hits_per_event * number_of_events * sizeof(uint)));

  // sbt
  // cudaCheck(cudaMalloc((void**)&dev_tracks_to_follow, number_of_events * TTF_MODULO * sizeof(uint)));
  dev_tracks_to_follow = dev_cluster_candidates;
  
  cudaCheck(cudaMalloc((void**)&dev_tracks, number_of_events * max_tracks_in_event * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_tracklets, average_number_of_hits_per_event * number_of_events * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_weak_tracks, average_number_of_hits_per_event * number_of_events * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_hit_used, average_number_of_hits_per_event * number_of_events * sizeof(bool)));
  cudaCheck(cudaMalloc((void**)&dev_atomics_storage, number_of_events * atomic_space * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_h0_candidates, 2 * average_number_of_hits_per_event * number_of_events * sizeof(short)));
  cudaCheck(cudaMalloc((void**)&dev_h2_candidates, 2 * average_number_of_hits_per_event * number_of_events * sizeof(short)));
  cudaCheck(cudaMalloc((void**)&dev_rel_indices, number_of_events * max_numhits_in_module * sizeof(unsigned short)));

  if (do_simplified_kalman_filter) {
    // simplified kalman filter
    cudaCheck(cudaMalloc((void**)&dev_velo_states, number_of_events * max_tracks_in_event * STATES_PER_TRACK * sizeof(VeloState)));
  }

  // Memory allocations for host memory (copy back)
  cudaCheck(cudaMallocHost((void**)&host_number_of_tracks_pinned, number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_tracks_pinned, number_of_events * max_tracks_in_event * sizeof(Track)));

  // Pre-populate raw_input data, in case the user requested -a 0
  cudaCheck(cudaMemcpyAsync(dev_raw_input, raw_events.data(), raw_events.size(), cudaMemcpyHostToDevice, stream));
  cudaCheck(cudaMemcpyAsync(dev_raw_input_offsets, event_offsets.data(), event_offsets.size() * sizeof(uint), cudaMemcpyHostToDevice, stream));

  // Prepare kernels
  estimateInputSize.set(
    estimate_input_size_blocks,
    estimate_input_size_threads,
    stream,
    dev_raw_input,
    dev_raw_input_offsets,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates
  );

  prefixSum.set(
    prefix_sum_blocks,
    prefix_sum_threads,
    stream,
    dev_estimated_input_size,
    number_of_events * 52
  );

  maskedVeloClustering.set(
    masked_velo_clustering_blocks,
    masked_velo_clustering_threads,
    stream,
    dev_raw_input,
    dev_raw_input_offsets,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates,
    dev_velo_cluster_container,
    dev_velo_geometry
  );

  calculatePhiAndSort.set(
    num_blocks,
    sort_num_threads,
    stream,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_velo_cluster_container,
    dev_hit_permutation
  );

  searchByTriplet.set(
    num_blocks,
    sbt_num_threads,
    stream,
    dev_velo_cluster_container,
    dev_estimated_input_size,
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

  consolidateTracks.set(
    consolidate_blocks,
    consolidate_num_threads,
    stream,
    dev_atomics_storage,
    dev_tracks,
    dev_tracklets
  );

  simplifiedKalmanFilter.set(
    num_blocks,
    simplified_kalman_num_threads,
    stream,
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_atomics_storage,
    dev_tracks,
    dev_velo_states,
    do_consolidate
  );

  return cudaSuccess;
}