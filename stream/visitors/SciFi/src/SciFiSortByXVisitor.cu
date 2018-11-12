#include "SequenceVisitor.cuh"
#include "SciFiSortByX.cuh"

template<>
void SequenceVisitor::set_arguments_size<scifi_sort_by_x_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_scifi_hit_permutations>(host_buffers.host_accumulated_number_of_scifi_hits[0]);
}

template<>
void SequenceVisitor::visit<scifi_sort_by_x_t>(
  scifi_sort_by_x_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(runtime_options.number_of_events), dim3(64), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_scifi_hits>(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.offset<dev_scifi_hit_permutations>()
  );

  state.invoke();
}
