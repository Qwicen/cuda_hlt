#include "StreamVisitor.cuh"

template<>
void StreamVisitor::visit<decltype(weak_tracks_adder_t(weak_tracks_adder))>(
  decltype(weak_tracks_adder_t(weak_tracks_adder))& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_scifi_hit_permutations>(host_buffers.host_accumulated_number_of_scifi_hits[0]);
  scheduler.setup_next(arguments, sequence_step);

  sequence.set_opts<seq::scifi_sort_by_x>(dim3(runtime_options.number_of_events), dim3(64), cuda_stream);
  sequence.set_arguments<seq::scifi_sort_by_x>(
    arguments.offset<arg::dev_scifi_hits>(),
    arguments.offset<arg::dev_scifi_hit_count>(),
    arguments.offset<arg::dev_scifi_hit_permutations>()
  );

  sequence.invoke<seq::scifi_sort_by_x>();
}
