#include "Stream.cuh"

/**
 * @brief Sets up statically the chain that will be
 *        executed later.
 *        
 * @details 
 */
cudaError_t Stream::initialize(
  const std::vector<char>& raw_events,
  const std::vector<uint>& event_offsets,
  const std::vector<char>& param_geometry,
  const uint number_of_events,
  const bool param_transmit_host_to_device,
  const bool param_transmit_device_to_host,
  const bool param_do_check,
  const bool param_do_simplified_kalman_filter,
  const bool param_print_individual_rates,
  const std::string& param_folder_name_MC,
  const size_t param_reserve_mb,
  const uint param_stream_number
) {
  // Set stream and events
  cudaCheck(cudaStreamCreate(&stream));
  cudaCheck(cudaEventCreate(&cuda_generic_event));
  cudaCheck(cudaEventCreate(&cuda_event_start));
  cudaCheck(cudaEventCreate(&cuda_event_stop));

  // Set stream options
  stream_number = param_stream_number;
  transmit_host_to_device = param_transmit_host_to_device;
  transmit_device_to_host = param_transmit_device_to_host;
  do_check = param_do_check;
  do_simplified_kalman_filter = param_do_simplified_kalman_filter;
  print_individual_rates = param_print_individual_rates;
  geometry = param_geometry;
  folder_name_MC = param_folder_name_MC;
  reserve_mb = param_reserve_mb;

  // Special case
  // Populate velo geometry
  cudaCheck(cudaMalloc((void**)&dev_velo_geometry, geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_velo_geometry, geometry.data(), geometry.size(), cudaMemcpyHostToDevice, stream));

  // Memory allocations for host memory (copy back)
  cudaCheck(cudaMallocHost((void**)&host_number_of_tracks, number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_tracks, number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hit_number, VeloTracking::max_tracks * number_of_events * sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hits, number_of_events * VeloTracking::max_tracks * 20 * sizeof(Hit<mc_check_enabled>)));
  cudaCheck(cudaMallocHost((void**)&host_total_number_of_velo_clusters, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_number_of_reconstructed_velo_tracks, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_hits_in_velo_tracks, sizeof(uint)));

  // Define sequence of algorithms to execute
  sequence = generate_sequence(
    generate_handler(estimate_input_size),
    generate_handler(prefix_sum_reduce),
    generate_handler(prefix_sum_single_block),
    generate_handler(prefix_sum_scan),
    generate_handler(masked_velo_clustering),
    generate_handler(calculatePhiAndSort),
    generate_handler(searchByTriplet),
    generate_handler(copy_and_prefix_sum_single_block),
    generate_handler(copy_velo_track_hit_number),
    generate_handler(prefix_sum_reduce),
    generate_handler(prefix_sum_single_block),
    generate_handler(prefix_sum_scan),
    generate_handler(consolidate_tracks)
  );

  // Set options for each algorithm
  // (number of blocks, number of threads, stream, dynamic shared memory space)
  prefixSumBlocks = (VeloTracking::n_modules * number_of_events + 511) / 512;
  prefixSumScanBlocks = prefixSumBlocks==1 ? 1 : (prefixSumBlocks-1);

  // Note: sequence.item can access the sequence by either the number
  //       of the sequence or the type of the kernel call itself
  //       (ie. sequence.item(estimate_input_size)). Doing the latter however
  //       may lead to compiler errors in the future if two kernels have
  //       the same signature
  sequence.item<seq::estimate_input_size>().set_opts(    dim3(number_of_events),    dim3(32, 26), stream);
  sequence.item<seq::prefix_sum_reduce>().set_opts(      dim3(prefixSumBlocks),     dim3(256),    stream);
  sequence.item<seq::prefix_sum_single_block>().set_opts(dim3(1),                   dim3(1024),   stream);
  sequence.item<seq::prefix_sum_scan>().set_opts(        dim3(prefixSumScanBlocks), dim3(512),    stream);
  sequence.item<seq::masked_velo_clustering>().set_opts( dim3(number_of_events),    dim3(256),    stream);
  sequence.item<seq::calculate_phi_and_sort>().set_opts( dim3(number_of_events),    dim3(64),     stream);
  sequence.item<seq::search_by_triplet>().set_opts(      dim3(number_of_events),    dim3(32),     stream, 32 * sizeof(float));
  sequence.item<seq::copy_and_prefix_sum_single_block>().set_opts(dim3(1),          dim3(1024),   stream);
  sequence.item<seq::copy_velo_track_hit_number>().set_opts(dim3(number_of_events), dim3(512),    stream);

  // Note: This item opt is not known at compile time
  // seq::prefix_sum_reduce_velo_track_hit_number

  sequence.item<seq::prefix_sum_single_block_velo_track_hit_number>().set_opts(dim3(1), dim3(1024), stream);

  // Note: Not known at compile time
  // seq::prefix_sum_scan_velo_track_hit_number

  sequence.item<seq::consolidate_tracks>().set_opts(     dim3(number_of_events),    dim3(32),     stream);
  // velo_kalman_filter_h.set_opts(dim3(number_of_events),    dim3(1024),   stream);

  // Define all arguments, with their type and size
  velo_cluster_container_size = number_of_events * VeloClustering::max_candidates_event * 2 * 6;
  size_t velo_states_size = number_of_events * VeloTracking::max_tracks;
  consolidate_tracks_average = VeloTracking::max_tracks;
  
  // size_t velo_states_size = do_simplified_kalman_filter ?
  //   number_of_events * VeloTracking::max_tracks * VeloTracking::states_per_track : 
  //   number_of_events * VeloTracking::max_tracks;

  // All arguments, with their type (without the *), name string, and size
  arguments = generate_tuple(
    Argument<char>{"dev_raw_input", raw_events.size()},
    Argument<uint>{"dev_raw_input_offsets", event_offsets.size()},
    Argument<uint>{"dev_estimated_input_size", (number_of_events * VeloTracking::n_modules + 2)},
    Argument<uint>{"dev_module_cluster_num", number_of_events * VeloTracking::n_modules},
    Argument<uint>{"dev_module_candidate_num", number_of_events},
    Argument<uint>{"dev_cluster_offset", number_of_events},
    Argument<uint>{"dev_cluster_candidates", number_of_events * VeloClustering::max_candidates_event},
    Argument<uint>{"dev_velo_cluster_container", velo_cluster_container_size},
    Argument<TrackHits>{"dev_tracks", number_of_events * VeloTracking::max_tracks},
    Argument<uint>{"dev_tracks_to_follow", number_of_events * VeloTracking::ttf_modulo},
    Argument<bool>{"dev_hit_used", VeloTracking::max_number_of_hits_per_event * number_of_events},
    Argument<int>{"dev_atomics_storage", number_of_events * VeloTracking::num_atomics},
    Argument<TrackHits>{"dev_tracklets", VeloTracking::ttf_modulo * number_of_events},
    Argument<uint>{"dev_weak_tracks", VeloTracking::ttf_modulo * number_of_events},
    Argument<short>{"dev_h0_candidates", 2 * VeloTracking::max_number_of_hits_per_event * number_of_events},
    Argument<short>{"dev_h2_candidates", 2 * VeloTracking::max_number_of_hits_per_event * number_of_events},
    Argument<unsigned short>{"dev_rel_indices", number_of_events * VeloTracking::max_numhits_in_module},
    Argument<uint>{"dev_hit_permutation", VeloTracking::max_number_of_hits_per_event * number_of_events},
    Argument<uint>{"dev_velo_track_hit_number", VeloTracking::max_tracks * number_of_events},
    Argument<uint>{"dev_prefix_sum_auxiliary_array_2", 0},
    Argument<Hit<mc_check_enabled>>{"dev_velo_track_hits", number_of_events * VeloTracking::max_tracks * 20},
    Argument<VeloState>{"dev_velo_states", velo_states_size}
  );
  
  // Fetch argument sizes
  // Note: The enum indices hold in the vector datatype
  std::vector<size_t> argument_sizes = generate_argument_sizes(arguments);
  std::vector<std::string> argument_names = generate_argument_names(arguments);

  // Set dependencies for each algorithm
  std::vector<std::vector<uint>> sequence_arguments
    (std::tuple_size<decltype(sequence.algorithms)>::value);

  sequence_arguments[seq::estimate_input_size] = {
    arg::dev_raw_input,
    arg::dev_raw_input_offsets,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_module_candidate_num,
    arg::dev_cluster_candidates
  };
  sequence_arguments[seq::prefix_sum_reduce] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_arguments[seq::prefix_sum_single_block] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_arguments[seq::prefix_sum_scan] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_arguments[seq::masked_velo_clustering] = {
    arg::dev_raw_input,
    arg::dev_raw_input_offsets,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_module_candidate_num,
    arg::dev_cluster_candidates,
    arg::dev_velo_cluster_container
  };
  sequence_arguments[seq::calculate_phi_and_sort] = {
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_velo_cluster_container,
    arg::dev_hit_permutation
  };
  sequence_arguments[seq::search_by_triplet] = {
    arg::dev_velo_cluster_container,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_tracks,
    arg::dev_tracklets,
    arg::dev_tracks_to_follow,
    arg::dev_weak_tracks,
    arg::dev_hit_used,
    arg::dev_atomics_storage,
    arg::dev_h0_candidates,
    arg::dev_h2_candidates,
    arg::dev_rel_indices
  };
  sequence_arguments[seq::copy_and_prefix_sum_single_block] = {
    arg::dev_atomics_storage
  };
  sequence_arguments[seq::copy_velo_track_hit_number] = {
    arg::dev_tracks,
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number
  };
  sequence_arguments[seq::prefix_sum_reduce_velo_track_hit_number] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_arguments[seq::prefix_sum_single_block_velo_track_hit_number] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_arguments[seq::prefix_sum_scan_velo_track_hit_number] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_arguments[seq::consolidate_tracks] = {
    arg::dev_atomics_storage,
    arg::dev_tracks,
    arg::dev_velo_track_hit_number,
    arg::dev_velo_cluster_container,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_velo_track_hits,
    arg::dev_velo_states
  };

  // Prepare dynamic scheduler
  scheduler = BaseDynamicScheduler{argument_names, sequence_arguments, reserve_mb * 1024 * 1024};

  // TODO: Malloc required GPU memory
  //       What is required memory here? Do we run the static scheduler?
  // For now, let's assume 1 GiB
  cudaCheck(cudaMalloc((void**)&dev_base_pointer, reserve_mb * 1024 * 1024));

  // velo_kalman_filter_h.set_arguments(
  //   dev_velo_cluster_container,
  //   dev_estimated_input_size,
  //   dev_atomics_storage,
  //   dev_velo_states
  // );

  return cudaSuccess;
}
