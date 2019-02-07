#include "SequenceVisitor.cuh" 
#include "EstimateInputSize.cuh"

template<>
void SequenceVisitor::set_arguments_size<velo_estimate_input_size_t>(
  velo_estimate_input_size_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  if (logger::ll.verbosityLevel >= logger::debug) {
    debug_cout << "# of events = " << host_buffers.host_number_of_selected_events[0] << std::endl;
  }
  
  arguments.set_size<dev_velo_raw_input>(runtime_options.host_velopix_events_size);
  arguments.set_size<dev_velo_raw_input_offsets>(runtime_options.host_velopix_event_offsets_size);
  arguments.set_size<dev_estimated_input_size>(host_buffers.host_number_of_selected_events[0] * Velo::Constants::n_modules + 1);
  arguments.set_size<dev_module_cluster_num>(host_buffers.host_number_of_selected_events[0] * Velo::Constants::n_modules);
  arguments.set_size<dev_module_candidate_num>(host_buffers.host_number_of_selected_events[0]);
  arguments.set_size<dev_cluster_candidates>(host_buffers.host_number_of_selected_events[0] * VeloClustering::max_candidates_event);
  arguments.set_size<dev_event_order>(host_buffers.host_number_of_selected_events[0]);
} 

template<>
void SequenceVisitor::visit<velo_estimate_input_size_t>(
  velo_estimate_input_size_t& state,
  const velo_estimate_input_size_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts and arguments for kernel call
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(32, 26), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_velo_raw_input>(),
    arguments.offset<dev_velo_raw_input_offsets>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_module_cluster_num>(),
    arguments.offset<dev_module_candidate_num>(),
    arguments.offset<dev_cluster_candidates>(),
    arguments.offset<dev_event_list>(),         
    arguments.offset<dev_event_order>(), 
    constants.dev_velo_candidate_ks
  );

  // Kernel call
  state.invoke();
}
