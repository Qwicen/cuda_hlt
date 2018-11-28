#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

template<>
void SequenceVisitor::set_arguments_size<copy_velo_track_hit_number_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_velo_track_hit_number>(host_buffers.velo_track_hit_number_size());
}

template<>
void SequenceVisitor::visit<copy_velo_track_hit_number_t>(
  copy_velo_track_hit_number_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(runtime_options.number_of_events), dim3(512), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_tracks>(),
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_velo_track_hit_number>()
  );

  state.invoke();
}