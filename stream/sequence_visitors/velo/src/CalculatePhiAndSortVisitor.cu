#include "SequenceVisitor.cuh"
#include "CalculatePhiAndSort.cuh"

template<>
void SequenceVisitor::visit<calculate_phi_and_sort_t>(
  calculate_phi_and_sort_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_hit_permutation>(host_buffers.host_total_number_of_velo_clusters[0]);
  scheduler.setup_next(arguments, sequence_step);
  state.set_opts(dim3(runtime_options.number_of_events), dim3(64), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_estimated_input_size>(),
    arguments.offset<arg::dev_module_cluster_num>(),
    arguments.offset<arg::dev_velo_cluster_container>(),
    arguments.offset<arg::dev_hit_permutation>()
  );
  state.invoke();
}
