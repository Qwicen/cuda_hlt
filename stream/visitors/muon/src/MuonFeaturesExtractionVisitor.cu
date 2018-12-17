#include "SequenceVisitor.cuh"
#include "MuonFeaturesExtraction.cuh"

template<>
void SequenceVisitor::set_arguments_size<muon_catboost_features_extraction_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{ 
  arguments.set_size<dev_muon_hits>(runtime_options.number_of_events);
  arguments.set_size<dev_muon_catboost_features>(constants.muon_catboost_n_features * host_buffers.host_number_of_reconstructed_scifi_tracks[0]);
}

template<>
void SequenceVisitor::visit<muon_catboost_features_extraction_t>(
  muon_catboost_features_extraction_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Copy memory from host to device
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_muon_hits>(),
    runtime_options.host_muon_hits_events.data(),
    runtime_options.number_of_events * sizeof(Muon::HitsSoA),
    cudaMemcpyHostToDevice,
    cuda_stream
  ));

  // Setup opts for kernel call
  state.set_opts(dim3(runtime_options.number_of_events, Muon::Constants::n_stations), dim3(32), cuda_stream);

  // Setup arguments for kernel call
  state.set_arguments(
    arguments.offset<dev_atomics_scifi>(),
    arguments.offset<dev_scifi_track_hit_number>(),
    arguments.offset<dev_scifi_qop>(),
    arguments.offset<dev_scifi_states>(),
    arguments.offset<dev_scifi_track_ut_indices>(),
    arguments.offset<dev_muon_hits>(),
    arguments.offset<dev_muon_catboost_features>()
  );

  // Kernel call
  state.invoke();
}
