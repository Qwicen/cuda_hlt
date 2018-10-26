#include "Stream.cuh"

/**
 * @brief Sets up the chain that will be executed later.
 */
cudaError_t Stream::initialize(
  const uint max_number_of_events,
  const bool param_do_check,
  const bool param_do_simplified_kalman_filter,
  const bool param_do_print_memory_manager,
  const bool param_run_on_x86,
  const std::string& param_folder_name_MC,
  const uint param_start_event_offset,
  const size_t reserve_mb,
  const uint param_stream_number,
  const Constants& param_constants
) {
  // Set stream and events
  cudaCheck(cudaStreamCreate(&cuda_stream));
  cudaCheck(cudaEventCreate(&cuda_generic_event));

  // Set stream options
  stream_number = param_stream_number;
  do_check = param_do_check;
  do_simplified_kalman_filter = param_do_simplified_kalman_filter;
  do_print_memory_manager = param_do_print_memory_manager;
  run_on_x86 = param_run_on_x86;
  folder_name_MC = param_folder_name_MC;
  start_event_offset = param_start_event_offset;
  constants = param_constants;

  // Reserve host buffers
  host_buffers.reserve(max_number_of_events);

  // Get dependencies for each algorithm
  std::vector<std::vector<int>> sequence_dependencies = get_sequence_dependencies();

  // Get output arguments from the sequence
  std::vector<int> sequence_output_arguments = get_sequence_output_arguments();

  // Prepare dynamic scheduler
  scheduler = {
    // get_sequence_names(),
    get_argument_names(), sequence_dependencies, sequence_output_arguments,
    reserve_mb * 1024 * 1024, do_print_memory_manager};

  // Malloc a configurable reserved memory
  cudaCheck(cudaMalloc((void**)&dev_base_pointer, reserve_mb * 1024 * 1024));

  return cudaSuccess;
}

cudaError_t Stream::run_sequence(const RuntimeOptions& runtime_options) {
  for (uint repetition=0; repetition<runtime_options.number_of_repetitions; ++repetition) {
    // Generate object for populating arguments
    ArgumentManager<argument_tuple_t> arguments {dev_base_pointer};

    // Reset scheduler
    scheduler.reset();

    // Visit all algorithms in configured sequence
    run_sequence_tuple(
      sequence_visitor,
      sequence_tuple,
      runtime_options,
      constants,
      arguments,
      scheduler,
      host_buffers,
      cuda_stream,
      cuda_generic_event);

    cudaEventRecord(cuda_generic_event, cuda_stream);
    cudaEventSynchronize(cuda_generic_event);
  }

  return cudaSuccess;
}

void Stream::run_monte_carlo_test(const uint number_of_events_requested) {
  std::cout << "Checking Velo tracks reconstructed on GPU" << std::endl;

  const std::vector<trackChecker::Tracks> tracks_events = prepareTracks(
    host_buffers.host_velo_tracks_atomics,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_track_hits,
    number_of_events_requested);

  call_pr_checker(
    tracks_events,
    folder_name_MC,
    start_event_offset,
    "Velo"
  );

  /* CHECKING VeloUT TRACKS */
  const std::vector<trackChecker::Tracks> veloUT_tracks = prepareVeloUTTracks(
    host_buffers.host_veloUT_tracks,
    host_buffers.host_atomics_veloUT,
    number_of_events_requested
  );

  std::cout << "Checking VeloUT tracks reconstructed on GPU" << std::endl;
  call_pr_checker(
    veloUT_tracks,
    folder_name_MC,
    start_event_offset,
    "VeloUT"
  );

  /* CHECKING Scifi TRACKS */
  const std::vector<trackChecker::Tracks> scifi_tracks = prepareForwardTracks(
    host_buffers.host_scifi_tracks,
    host_buffers.host_n_scifi_tracks,
    number_of_events_requested
  );
  
  std::cout << "Checking SciFi tracks reconstructed on GPU" << std::endl;
  call_pr_checker (
    scifi_tracks,
    folder_name_MC,
    start_event_offset,
    "Forward");
}
