#include "SequenceVisitor.cuh"
#include "MuonFeaturesExtraction.cuh"

template<>
void SequenceVisitor::set_arguments_size<muon_catboost_features_extraction_t>(
  muon_catboost_features_extraction_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_muon_catboost_features>(
    Muon::Constants::n_catboost_features * host_buffers.host_number_of_reconstructed_scifi_tracks[0]);
}

template<>
void SequenceVisitor::visit<muon_catboost_features_extraction_t>(
  muon_catboost_features_extraction_t& state,
  const muon_catboost_features_extraction_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts for kernel call
  state.set_opts(
    dim3(host_buffers.host_number_of_selected_events[0], Muon::Constants::n_stations), dim3(32), cuda_stream);

  // Setup arguments for kernel call
  state.set_arguments(
    arguments.offset<dev_atomics_scifi>(),
    arguments.offset<dev_scifi_track_hit_number>(),
    arguments.offset<dev_scifi_qop>(),
    arguments.offset<dev_scifi_states>(),
    arguments.offset<dev_scifi_track_ut_indices>(),
    arguments.offset<dev_muon_hits>(),
    arguments.offset<dev_muon_catboost_features>());

  // Kernel call
  state.invoke();
}
