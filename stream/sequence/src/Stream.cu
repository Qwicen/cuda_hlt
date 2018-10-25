#include "Stream.cuh"

/**
 * @brief Sets up the chain that will be executed later.
 */
cudaError_t Stream::initialize(
  const std::vector<char>& velopix_geometry,
  const std::vector<char>& ut_boards,
  const std::vector<char>& ut_geometry,
  const std::vector<char>& ut_magnet_tool,
  const std::vector<char>& scifi_geometry,
  const uint max_number_of_events,
  const bool param_do_check,
  const bool param_do_simplified_kalman_filter,
  const bool param_do_print_memory_manager,
  const bool param_run_on_x86,
  const bool param_data_preparation_only,
  const std::string& param_folder_name_MC,
  const uint param_start_event_offset,
  const size_t reserve_mb,
  const uint param_stream_number,
  const Constants& param_constants
) {
  // Set stream and events
  cudaCheck(cudaStreamCreate(&stream));
  cudaCheck(cudaEventCreate(&cuda_generic_event));
  cudaCheck(cudaEventCreate(&cuda_event_start));
  cudaCheck(cudaEventCreate(&cuda_event_stop));

  // Set stream options
  stream_number = param_stream_number;
  do_check = param_do_check;
  do_simplified_kalman_filter = param_do_simplified_kalman_filter;
  do_print_memory_manager = param_do_print_memory_manager;
  run_on_x86 = param_run_on_x86;
  data_preparation_only = param_data_preparation_only;
  folder_name_MC = param_folder_name_MC;
  start_event_offset = param_start_event_offset;
  constants = param_constants;

  // Special case
  // Populate velo geometry
  cudaCheck(cudaMalloc((void**)&dev_velo_geometry, velopix_geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_velo_geometry, velopix_geometry.data(), velopix_geometry.size(), cudaMemcpyHostToDevice, stream));

  // Populate UT boards and geometry
  cudaCheck(cudaMalloc((void**)&dev_ut_boards, ut_boards.size()));
  cudaCheck(cudaMemcpyAsync(dev_ut_boards, ut_boards.data(), ut_boards.size(), cudaMemcpyHostToDevice, stream));

  cudaCheck(cudaMalloc((void**)&dev_ut_geometry, ut_geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_ut_geometry, ut_geometry.data(), ut_geometry.size(), cudaMemcpyHostToDevice, stream));

  // Populate UT magnet tool values
  cudaCheck(cudaMalloc((void**)&dev_ut_magnet_tool, ut_magnet_tool.size()));
  cudaCheck(cudaMemcpyAsync(dev_ut_magnet_tool, ut_magnet_tool.data(), ut_magnet_tool.size(), cudaMemcpyHostToDevice, stream));

  // Populate FT geometry
  cudaCheck(cudaMalloc((void**)&dev_scifi_geometry, scifi_geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_scifi_geometry, scifi_geometry.data(), scifi_geometry.size(), cudaMemcpyHostToDevice, stream));

  // Memory allocations for host memory (copy back)
  cudaCheck(cudaMallocHost((void**)&host_velo_tracks_atomics, (2 * max_number_of_events + 1) * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hit_number, max_number_of_events * VeloTracking::max_tracks * sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hits, max_number_of_events * VeloTracking::max_tracks * VeloTracking::max_track_size * sizeof(Velo::Hit)));
  cudaCheck(cudaMallocHost((void**)&host_total_number_of_velo_clusters, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_number_of_reconstructed_velo_tracks, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_hits_in_velo_tracks, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_velo_states, max_number_of_events * VeloTracking::max_tracks * sizeof(Velo::State)));
  cudaCheck(cudaMallocHost((void**)&host_veloUT_tracks, max_number_of_events * VeloUTTracking::max_num_tracks * sizeof(VeloUTTracking::TrackUT)));
  cudaCheck(cudaMallocHost((void**)&host_atomics_veloUT, VeloUTTracking::num_atomics * max_number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_ut_hits, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_scifi_tracks, max_number_of_events * SciFi::max_tracks * sizeof(SciFi::Track)));
  cudaCheck(cudaMallocHost((void**)&host_n_scifi_tracks, max_number_of_events * sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_scifi_hits, sizeof(uint)));

  // Define sequence of algorithms to execute
  sequence.set(sequence_algorithms());

  // Set options for each algorithm
  // (number of blocks, number of threads, stream, dynamic shared memory space)
  // Setup sequence items opts that are static and will not change
  // regardless of events on flight
  sequence.set_opts<seq::prefix_sum_single_block>(                      dim3(1), dim3(1024), stream);
  sequence.set_opts<seq::copy_and_prefix_sum_single_block>(             dim3(1), dim3(1024), stream);
  sequence.set_opts<seq::prefix_sum_single_block_velo_track_hit_number>(dim3(1), dim3(1024), stream);
  sequence.set_opts<seq::prefix_sum_single_block_ut_hits>(              dim3(1), dim3(1024), stream);
  sequence.set_opts<seq::prefix_sum_single_block_scifi_hits>(           dim3(1), dim3(1024), stream);

  // Get dependencies for each algorithm
  std::vector<std::vector<int>> sequence_dependencies = get_sequence_dependencies();

  // Get output arguments from the sequence
  std::vector<int> sequence_output_arguments = get_sequence_output_arguments();

  // Prepare dynamic scheduler
  scheduler = {get_sequence_names(), get_argument_names(),
    sequence_dependencies, sequence_output_arguments,
    reserve_mb * 1024 * 1024, do_print_memory_manager};

  // Malloc a configurable reserved memory
  cudaCheck(cudaMalloc((void**)&dev_base_pointer, reserve_mb * 1024 * 1024));

  return cudaSuccess;
}

void Stream::print_timing(
  const unsigned int number_of_events,
  const std::vector<std::pair<std::string, float>>& times
) {
  const auto total_time = times[times.size() - 1];
  std::string partial_times = "{\n";
  for (size_t i=0; i<times.size(); ++i) {
    if (i != times.size()-1) {
      partial_times += " " + times[i].first + "\t" + std::to_string(times[i].second) + "\t("
        + std::to_string(100 * (times[i].second / total_time.second)) + " %)\n";
    } else {
      partial_times += " " + times[i].first + "\t" + std::to_string(times[i].second) + "\t("
        + std::to_string(100 * (times[i].second / total_time.second)) + " %)\n}";
    }
  }

  info_cout << "stream #" << stream_number << ": "
    << number_of_events / total_time.second << " events/s"
    << ", partial timers (s): " << partial_times
    << std::endl;
}
