#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

template<>
void SequenceVisitor::set_arguments_size<copy_ut_track_hit_number_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_ut_track_hit_number>(host_buffers.ut_track_hit_number_size());
}

template<>
void SequenceVisitor::visit<copy_ut_track_hit_number_t>(
  copy_ut_track_hit_number_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(512), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_ut_tracks>(),
    arguments.offset<dev_atomics_ut>(),
    arguments.offset<dev_ut_track_hit_number>()
  );

  state.invoke();
}
