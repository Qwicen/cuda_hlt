#include "SequenceVisitor.cuh"
#include "MuonFeaturesExtraction.cuh"

template<>
void SequenceVisitor::set_arguments_size<muon_catboost_features_extraction_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{ 
  arguments.set_size<dev_scifi_states>(host_buffers.host_number_of_reconstructed_scifi_tracks[0]);
  arguments.set_size<dev_muon_hits>(1);
  arguments.set_size<dev_muon_catboost_features>(constants.muon_catboost_n_features);
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
    &runtime_options.host_muon_hits_events[0],
    1 * sizeof(Muon::HitsSoA), //todo
    cudaMemcpyHostToDevice,
    cuda_stream
  ));

  // Setup opts for kernel call
  state.set_opts(dim3(Muon::Constants::n_stations), dim3(1), cuda_stream);

  // Setup arguments for kernel call
  state.set_arguments(
    arguments.offset<dev_scifi_states>(),
    arguments.offset<dev_muon_hits>(),
    arguments.offset<dev_muon_catboost_features>()
  );

  // Kernel call
  state.invoke();

  // Retrieve result
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_muon_catboost_features,
    arguments.offset<dev_muon_catboost_features>(),
    arguments.size<dev_muon_catboost_features>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));

  // Wait to receive the result
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // Check the output
  debug_cout << "MUON FEATURES: " << std::endl;
  for (int i = 0; i < constants.muon_catboost_n_features; i++) {
    debug_cout << host_buffers.host_muon_catboost_features[i] << " ";
  }
  debug_cout << std::endl << std::endl;
}
