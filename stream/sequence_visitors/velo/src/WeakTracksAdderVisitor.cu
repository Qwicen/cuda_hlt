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
  scheduler.setup_next(arguments, sequence_step);

  // Setup opts and arguments
  state.set_opts(dim3(runtime_options.number_of_events), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_velo_cluster_container>(),
    arguments.offset<arg::dev_estimated_input_size>(),
    arguments.offset<arg::dev_tracks>(),
    arguments.offset<arg::dev_weak_tracks>(),
    arguments.offset<arg::dev_hit_used>(),
    arguments.offset<arg::dev_atomics_storage>()
  );
  
  state.invoke();
}
