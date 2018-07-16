#include "Stream.cuh"

/**
 * @brief Sets up the chain that will be executed later.
 */
cudaError_t Stream::initialize(
  const std::vector<char>& velopix_geometry,
  const PrUTMagnetTool* host_ut_magnet_tool,
  const uint max_number_of_events,
  const bool param_transmit_device_to_host,
  const bool param_do_check,
  const bool param_do_simplified_kalman_filter,
  const bool param_do_print_memory_manager,
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
  transmit_device_to_host = param_transmit_device_to_host;
  do_check = param_do_check;
  do_simplified_kalman_filter = param_do_simplified_kalman_filter;
  do_print_memory_manager = param_do_print_memory_manager;
  folder_name_MC = param_folder_name_MC;

  // Special case
  // Populate velo geometry
  cudaCheck(cudaMalloc((void**)&dev_velo_geometry, velopix_geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_velo_geometry, velopix_geometry.data(), velopix_geometry.size(), cudaMemcpyHostToDevice, stream));

  // Populate UT magnet tool values
  cudaCheck(cudaMalloc((void**)&dev_ut_magnet_tool, sizeof(PrUTMagnetTool)));
  cudaCheck(cudaMemcpyAsync(dev_ut_magnet_tool, host_ut_magnet_tool, sizeof(PrUTMagnetTool), cudaMemcpyHostToDevice, stream));
  // copy to constant memory variables, defined in PrVeloUTDefinitions.cuh 
  cudaCheck(cudaMemcpyToSymbol(VeloUTTracking::dev_dxDyTable, VeloUTTracking::dxDyTable, VeloUTTracking::n_layers * sizeof(float)));
  float(cudaMemcpyToSymbol(PrVeloUTConst::dev_minValsBdl, PrVeloUTConst::minValsBdl, 3 * sizeof(float)));
  cudaCheck(cudaMemcpyToSymbol(PrVeloUTConst::dev_maxValsBdl, PrVeloUTConst::maxValsBdl, 3 * sizeof(float)));
  cudaCheck(cudaMemcpyToSymbol(PrVeloUTConst::dev_deltaBdl, PrVeloUTConst::deltaBdl, 3 * sizeof(float)));
  cudaCheck(cudaMemcpyToSymbol(PrVeloUTConst::dev_dxDyHelper, PrVeloUTConst::dxDyHelper, 4 * sizeof(float)));
  
  // Memory allocations for host memory (copy back)
  cudaCheck(cudaMallocHost((void**)&host_number_of_tracks, max_number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_tracks, max_number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hit_number, max_number_of_events * VeloTracking::max_tracks * sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_velo_track_hits, max_number_of_events * VeloTracking::max_tracks * 20 * sizeof(VeloTracking::Hit<mc_check_enabled>)));
  cudaCheck(cudaMallocHost((void**)&host_total_number_of_velo_clusters, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_number_of_reconstructed_velo_tracks, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_number_of_hits_in_velo_tracks, sizeof(uint)));
  cudaCheck(cudaMallocHost((void**)&host_velo_states, max_number_of_events * VeloTracking::max_tracks * sizeof(VeloState)));
  cudaCheck(cudaMallocHost((void**)&host_veloUT_tracks, max_number_of_events * VeloUTTracking::max_num_tracks * sizeof(VeloUTTracking::TrackUT)));

  // Define sequence of algorithms to execute
  sequence.set(sequence_algorithms());

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

  // Get dependencies for each algorithm
  std::vector<std::vector<int>> sequence_dependencies = get_sequence_dependencies();

  // Get output arguments from the sequence
  std::vector<int> sequence_output_arguments = get_sequence_output_arguments();

  // Prepare dynamic scheduler
  scheduler = BaseDynamicScheduler{sequence_names, argument_names,
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
