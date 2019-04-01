#include "SequenceVisitor.cuh"
#include "IsMuon.cuh"

template<>
void SequenceVisitor::set_arguments_size<is_muon_t>(
  is_muon_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{ 
  arguments.set_size<dev_muon_hits>(host_buffers.host_max_number_of_events);
  arguments.set_size<dev_muon_track_occupancies>(
    Muon::Constants::n_stations * host_buffers.host_number_of_reconstructed_scifi_tracks[0]
  );
  arguments.set_size<dev_is_muon>(host_buffers.host_number_of_reconstructed_scifi_tracks[0]);
}

template<>
void SequenceVisitor::visit<is_muon_t>(
  is_muon_t& state,
  const is_muon_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Copy memory from host to device
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_muon_hits>(),
    runtime_options.host_muon_hits_events.data(),
    host_buffers.host_max_number_of_events * sizeof(Muon::HitsSoA),
    cudaMemcpyHostToDevice,
    cuda_stream
  ));

  // Setup opts for kernel call
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(32, Muon::Constants::n_stations), cuda_stream);

  // Setup arguments for kernel call
  state.set_arguments(
    arguments.offset<dev_atomics_scifi>(),
    arguments.offset<dev_scifi_track_hit_number>(),
    arguments.offset<dev_scifi_qop>(),
    arguments.offset<dev_scifi_states>(),
    arguments.offset<dev_scifi_track_ut_indices>(),
    arguments.offset<dev_muon_hits>(),
    arguments.offset<dev_muon_track_occupancies>(),
    arguments.offset<dev_is_muon>(),
    arguments.offset<dev_event_list>(),
    constants.dev_muon_foi,
    constants.dev_muon_momentum_cuts
  );

  // Kernel call
  state.invoke();

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_is_muon,
    arguments.offset<dev_is_muon>(),
    arguments.size<dev_is_muon>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));
}
