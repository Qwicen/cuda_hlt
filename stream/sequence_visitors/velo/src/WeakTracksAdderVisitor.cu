#include "SequenceVisitor.cuh"
#include "WeakTracksAdder.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(weak_tracks_adder_t)

template<>
void SequenceVisitor::visit<weak_tracks_adder_t>(
  weak_tracks_adder_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts and arguments
  state.set_opts(dim3(runtime_options.number_of_events), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_velo_cluster_container>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_tracks>(),
    arguments.offset<dev_weak_tracks>(),
    arguments.offset<dev_hit_used>(),
    arguments.offset<dev_atomics_storage>()
  );
  
  state.invoke();
}
