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
  const std::string& param_folder_name_pv,
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
  folder_name_pv = param_folder_name_pv;
  start_event_offset = param_start_event_offset;
  constants = param_constants;


  // Reserve host buffers
  host_buffers.reserve(max_number_of_events);


  // Malloc a configurable reserved memory
  cudaCheck(cudaMalloc((void**)&dev_base_pointer, reserve_mb * 1024 * 1024));

  // Prepare scheduler
  scheduler = {
    do_print_memory_manager,
    reserve_mb * 1024 * 1024,
    dev_base_pointer
  };

  return cudaSuccess;
}

cudaError_t Stream::run_sequence(const RuntimeOptions& runtime_options) {
  for (uint repetition=0; repetition<runtime_options.number_of_repetitions; ++repetition) {
    // Reset scheduler
    scheduler.reset();

    // Visit all algorithms in configured sequence
    Sch::RunSequenceTuple<
      scheduler_t,
      SequenceVisitor,
      configured_sequence_t,
      std::tuple<
        const RuntimeOptions&,
        const Constants&,
        const HostBuffers&,
        argument_manager_t&
      >,
      std::tuple<
        const RuntimeOptions&,
        const Constants&,
        argument_manager_t&,
        HostBuffers&,
        cudaStream_t&,
        cudaEvent_t&
      >
    >::run(
      scheduler,
      sequence_visitor,
      sequence_tuple,
      // Arguments to set_arguments_size
      runtime_options,
      constants,
      host_buffers,
      scheduler.arguments(),
      // Arguments to visit
      runtime_options,
      constants,
      scheduler.arguments(),
      host_buffers,
      cuda_stream,
      cuda_generic_event);

    // Synchronize CUDA device
    cudaEventRecord(cuda_generic_event, cuda_stream);
    cudaEventSynchronize(cuda_generic_event);
  }

  return cudaSuccess;
}

void Stream::run_monte_carlo_test(const uint number_of_events_requested) {
  std::cout << "Checking Velo tracks reconstructed on GPU" << std::endl;

#ifdef WITH_ROOT
  TFile *f = new TFile("../output/PrCheckerPlots.root", "RECREATE");
  f->Close();
#endif
  
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
