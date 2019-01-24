#include "SequenceVisitor.cuh"
#include "IsMuon.cuh"

template<>
void SequenceVisitor::set_arguments_size<is_muon_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{ 
  arguments.set_size<dev_is_muon>(host_buffers.host_number_of_reconstructed_scifi_tracks[0]);
}

template<>
void SequenceVisitor::visit<is_muon_t>(
  is_muon_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts for kernel call
  state.set_opts(1, 1, cuda_stream);

  // Setup arguments for kernel call
  state.set_arguments(
    arguments.offset<dev_atomics_scifi>(),
    arguments.offset<dev_scifi_track_hit_number>(),
    arguments.offset<dev_scifi_qop>(),
    arguments.offset<dev_scifi_states>(),
    arguments.offset<dev_scifi_track_ut_indices>(),
    arguments.offset<dev_is_muon>(),
    constants.dev_muon_foi,
    constants.dev_muon_momentum_cuts
  );

  // Kernel call
  state.invoke();
}
