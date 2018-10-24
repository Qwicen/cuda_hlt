#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

template<>
void SequenceVisitor::visit<copy_velo_track_hit_number_t>(
  copy_velo_track_hit_number_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_velo_track_hit_number>(host_buffers.velo_track_hit_number_size());
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events), dim3(512), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_tracks>(),
    arguments.offset<arg::dev_atomics_storage>(),
    arguments.offset<arg::dev_velo_track_hit_number>()
  );

  state.invoke();
}
