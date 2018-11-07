#include "SequenceVisitor.cuh"
#include "FillCandidates.cuh"

template<>
void SequenceVisitor::visit<fill_candidates_t>(
  fill_candidates_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_h0_candidates>(2 * host_buffers.host_total_number_of_velo_clusters[0]);
  arguments.set_size<arg::dev_h2_candidates>(2 * host_buffers.host_total_number_of_velo_clusters[0]);
  scheduler.setup_next(arguments, sequence_step);
  
  // Setup opts and arguments
  state.set_opts(dim3(runtime_options.number_of_events, 48), dim3(128), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_velo_cluster_container>(),
    arguments.offset<arg::dev_estimated_input_size>(),
    arguments.offset<arg::dev_module_cluster_num>(),
    arguments.offset<arg::dev_h0_candidates>(),
    arguments.offset<arg::dev_h2_candidates>()
  );
  state.invoke();
}
