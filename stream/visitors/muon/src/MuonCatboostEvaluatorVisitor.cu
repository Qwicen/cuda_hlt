#include "SequenceVisitor.cuh"
#include "MuonCatboostEvaluator.cuh"
#include <vector>

template<>
void SequenceVisitor::set_arguments_size<muon_catboost_evaluator_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_muon_catboost_output>(host_buffers.host_number_of_reconstructed_scifi_tracks[0]);
}

template<>
void SequenceVisitor::visit<muon_catboost_evaluator_t>(
  muon_catboost_evaluator_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_reconstructed_scifi_tracks[0]),dim3(32), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_muon_catboost_features>(),
    arguments.offset<dev_muon_catboost_output>(),
    constants.dev_muon_catboost_leaf_values,
    constants.dev_muon_catboost_leaf_offsets,
    constants.dev_muon_catboost_split_borders,
    constants.dev_muon_catboost_split_features,
    constants.dev_muon_catboost_tree_depths,
    constants.dev_muon_catboost_tree_offsets,
    constants.muon_catboost_n_trees
  );
  state.invoke();
  
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_muon_catboost_output,
    arguments.offset<dev_muon_catboost_output>(),
    arguments.size<dev_muon_catboost_output>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
  debug_cout << "IsMuon" << std::endl;
  for(int i = 0; i < host_buffers.host_number_of_reconstructed_scifi_tracks[0]; ++i) {
    debug_cout << host_buffers.host_muon_catboost_output[i] << std::endl;
  }
}
