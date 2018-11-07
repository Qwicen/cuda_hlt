#include "SequenceVisitor.cuh"
#include "EstimateInputSize.cuh"

template<>
void SequenceVisitor::visit<estimate_input_size_t>(
  estimate_input_size_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Set arguments and reserve memory
  arguments.set_size<arg::dev_raw_input>(runtime_options.host_velopix_events_size);
  arguments.set_size<arg::dev_raw_input_offsets>(runtime_options.host_velopix_event_offsets_size);
  arguments.set_size<arg::dev_estimated_input_size>(runtime_options.number_of_events * VeloTracking::n_modules + 1);
  arguments.set_size<arg::dev_module_cluster_num>(runtime_options.number_of_events * VeloTracking::n_modules);
  arguments.set_size<arg::dev_module_candidate_num>(runtime_options.number_of_events);
  arguments.set_size<arg::dev_cluster_candidates>(runtime_options.number_of_events * VeloClustering::max_candidates_event);
  scheduler.setup_next(arguments, sequence_step);

  // Setup opts and arguments for kernel call
  state.set_opts(dim3(runtime_options.number_of_events), dim3(32, 26), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_raw_input>(),
    arguments.offset<arg::dev_raw_input_offsets>(),
    arguments.offset<arg::dev_estimated_input_size>(),
    arguments.offset<arg::dev_module_cluster_num>(),
    arguments.offset<arg::dev_module_candidate_num>(),
    arguments.offset<arg::dev_cluster_candidates>(),
    constants.dev_velo_candidate_ks
  );

  // Fetch required arguments
  cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_raw_input>(), runtime_options.host_velopix_events, arguments.size<arg::dev_raw_input>(), cudaMemcpyHostToDevice, cuda_stream));
  cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_raw_input_offsets>(), runtime_options.host_velopix_event_offsets, arguments.size<arg::dev_raw_input_offsets>(), cudaMemcpyHostToDevice, cuda_stream));
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // Kernel call
  state.invoke();
}
