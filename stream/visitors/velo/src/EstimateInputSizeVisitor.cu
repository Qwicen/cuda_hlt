#include "SequenceVisitor.cuh" 
#include "EstimateInputSize.cuh"

template<>
void SequenceVisitor::set_arguments_size<velo_estimate_input_size_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_estimated_input_size>(runtime_options.number_of_events * Velo::Constants::n_modules + 1);
  arguments.set_size<dev_module_cluster_num>(runtime_options.number_of_events * Velo::Constants::n_modules);
  arguments.set_size<dev_module_candidate_num>(runtime_options.number_of_events);
  arguments.set_size<dev_cluster_candidates>(runtime_options.number_of_events * VeloClustering::max_candidates_event);
} 

template<>
void SequenceVisitor::visit<velo_estimate_input_size_t>(
  velo_estimate_input_size_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{ 
  // Setup opts and arguments for kernel call
  state.set_opts(dim3(runtime_options.number_of_events), dim3(32, 26), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_raw_input>(),
    arguments.offset<dev_raw_input_offsets>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_module_cluster_num>(),
    arguments.offset<dev_module_candidate_num>(),
    arguments.offset<dev_cluster_candidates>(),
    constants.dev_velo_candidate_ks
  );

  // Kernel call
  state.invoke();
}
