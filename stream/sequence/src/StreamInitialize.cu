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
  const std::vector<char>& geometry,
  const uint max_number_of_events,
  const bool param_transmit_host_to_device,
  const bool param_transmit_device_to_host,
  const bool param_do_check,
  const bool param_do_simplified_kalman_filter,
  const bool param_print_individual_rates,
  const std::string& param_folder_name_MC,
  const size_t reserve_mb,
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
  folder_name_MC = param_folder_name_MC;

  // Special case
  // Populate velo geometry
  cudaCheck(cudaMalloc((void**)&dev_velo_geometry, geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_velo_geometry, geometry.data(), geometry.size(), cudaMemcpyHostToDevice, stream));

  // Memory allocations for host memory (copy back)
  cudaCheck(cudaMallocHost((void**)&host_number_of_tracks, max_number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_tracks, max_number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hit_number, max_number_of_events * VeloTracking::max_tracks * sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hits, max_number_of_events * VeloTracking::max_tracks * 20 * sizeof(Hit<mc_check_enabled>)));
  cudaCheck(cudaMallocHost((void**)&host_total_number_of_velo_clusters, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_number_of_reconstructed_velo_tracks, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_hits_in_velo_tracks, sizeof(uint)));

  // Define sequence of algorithms to execute
  sequence.set(
    estimate_input_size,
    prefix_sum_reduce,
    prefix_sum_single_block,
    prefix_sum_scan,
    masked_velo_clustering,
    calculatePhiAndSort,
    searchByTriplet,
    copy_and_prefix_sum_single_block,
    copy_velo_track_hit_number,
    prefix_sum_reduce,
    prefix_sum_single_block,
    prefix_sum_scan,
    consolidate_tracks
  );

  // Get sequence and argument names
  sequence_names = get_sequence_names();
  argument_names = get_argument_names();

  // Set options for each algorithm
  // (number of blocks, number of threads, stream, dynamic shared memory space)
  // Setup sequence items opts that are static and will not change
  // regardless of events on flight
  sequence.item<seq::prefix_sum_single_block>().set_opts(                      dim3(1), dim3(1024), stream);
  sequence.item<seq::copy_and_prefix_sum_single_block>().set_opts(             dim3(1), dim3(1024), stream);
  sequence.item<seq::prefix_sum_single_block_velo_track_hit_number>().set_opts(dim3(1), dim3(1024), stream);

  // Set dependencies for each algorithm
  std::vector<std::vector<uint>> sequence_dependencies (std::tuple_size<argument_tuple_t>::value);

  sequence_dependencies[seq::estimate_input_size] = {
    arg::dev_raw_input,
    arg::dev_raw_input_offsets,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_module_candidate_num,
    arg::dev_cluster_candidates
  };
  sequence_dependencies[seq::prefix_sum_reduce] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_dependencies[seq::prefix_sum_single_block] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_dependencies[seq::prefix_sum_scan] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_dependencies[seq::masked_velo_clustering] = {
    arg::dev_raw_input,
    arg::dev_raw_input_offsets,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_module_candidate_num,
    arg::dev_cluster_candidates,
    arg::dev_velo_cluster_container
  };
  sequence_dependencies[seq::calculate_phi_and_sort] = {
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_velo_cluster_container,
    arg::dev_hit_permutation
  };
  sequence_dependencies[seq::search_by_triplet] = {
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
  sequence_dependencies[seq::copy_and_prefix_sum_single_block] = {
    arg::dev_atomics_storage
  };
  sequence_dependencies[seq::copy_velo_track_hit_number] = {
    arg::dev_tracks,
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number
  };
  sequence_dependencies[seq::prefix_sum_reduce_velo_track_hit_number] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_dependencies[seq::prefix_sum_single_block_velo_track_hit_number] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_dependencies[seq::prefix_sum_scan_velo_track_hit_number] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_dependencies[seq::consolidate_tracks] = {
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
  scheduler = BaseDynamicScheduler{sequence_names, argument_names,
    sequence_dependencies, reserve_mb * 1024 * 1024};

  // Malloc a configurable reserved memory
  cudaCheck(cudaMalloc((void**)&dev_base_pointer, reserve_mb * 1024 * 1024));

  return cudaSuccess;
}
