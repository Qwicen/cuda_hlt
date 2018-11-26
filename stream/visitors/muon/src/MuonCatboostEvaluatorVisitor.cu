#include "SequenceVisitor.cuh"
#include "MuonCatboostEvaluator.cuh"
#include<vector>

template<>
void SequenceVisitor::set_arguments_size<muon_catboost_evaluator_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{ 
  int event_N = 1;
  arguments.set_size<dev_muon_catboost_output>(event_N);
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
  int event_N = 1;
  state.set_opts(dim3(event_N), dim3(32), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_muon_catboost_features>(),
    constants.dev_muon_catboost_leaf_values,
    constants.dev_muon_catboost_leaf_offsets,
    constants.dev_muon_catboost_split_borders,
    constants.dev_muon_catboost_split_features,
    constants.dev_muon_catboost_tree_depths,
    constants.dev_muon_catboost_tree_offsets,
    constants.muon_catboost_n_trees,
    constants.muon_catboost_n_features,
    event_N,
    arguments.offset<dev_muon_catboost_output>()
  );
  state.invoke();
  std::vector<float> output(event_N);
  
  cudaCheck(cudaMemcpyAsync(
    output.data(),
    arguments.offset<dev_muon_catboost_output>(),
    arguments.size<dev_muon_catboost_output>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
  debug_cout << "IsMuon" << std::endl;
  for(int i = 0; i < event_N; ++i) {
    debug_cout << output[i] << std::endl;
  }
}
