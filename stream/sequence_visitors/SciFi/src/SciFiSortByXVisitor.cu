#include "SequenceVisitor.cuh"
#include "SciFiSortByX.cuh"

template<>
void SequenceVisitor::visit<scifi_sort_by_x_t>(
  scifi_sort_by_x_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_scifi_hit_permutations>(host_buffers.host_accumulated_number_of_scifi_hits[0]);
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events), dim3(64), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_scifi_hits>(),
    arguments.offset<arg::dev_scifi_hit_count>(),
    arguments.offset<arg::dev_scifi_hit_permutations>()
  );

  state.invoke();
}
