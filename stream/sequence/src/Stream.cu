#include "Stream.cuh"

// Include the sequence checker specializations
#include "VeloSequenceCheckers_impl.cuh"
#include "UTSequenceCheckers_impl.cuh"
#include "SciFiSequenceCheckers_impl.cuh"
#include "PVSequenceCheckers_impl.cuh"

// For checking kalman filter.
#include "ParKalmanDefinitions.cuh"
#include "KalmanSequenceCheckers_impl.cuh"

/**
 * @brief Sets up the chain that will be executed later.
 */
cudaError_t Stream::initialize(
  const uint max_number_of_events,
  const bool param_do_print_memory_manager,
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
  do_print_memory_manager = param_do_print_memory_manager;
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
    // Initialize selected_number_of_events with requested_number_of_events
    host_buffers.host_number_of_selected_events[0] = runtime_options.number_of_events;

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

void Stream::run_monte_carlo_test(
  const std::string& mc_folder,
  const uint number_of_events_requested)
{
#ifdef WITH_ROOT
  TFile *f = new TFile("../output/PrCheckerPlots.root", "RECREATE");
  f->Close();
#endif

  // Create the CheckerInvoker and read Monte Carlo validation information
  const auto checker_invoker = CheckerInvoker(
    mc_folder,
    start_event_offset,
    host_buffers.host_event_list,
    number_of_events_requested,
    host_buffers.host_number_of_selected_events[0]);

  Sch::RunChecker<
    SequenceVisitor,
    configured_sequence_t,
    std::tuple<
      const uint&,
      const uint&,
      const HostBuffers&,
      const Constants&,
      const CheckerInvoker&
    >
  >::check(
    sequence_visitor,
    start_event_offset,
    number_of_events_requested,
    host_buffers,
    constants,
    checker_invoker
  );
}
