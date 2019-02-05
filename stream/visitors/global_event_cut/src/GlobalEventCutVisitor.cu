#include "SequenceVisitor.cuh" 
#include "GlobalEventCut.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(global_event_cut_t)

template<>
void SequenceVisitor::visit<global_event_cut_t>(
  global_event_cut_t& state,
  const global_event_cut_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{ 
  host_buffers.host_number_of_selected_events[0] = 0;

  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_number_of_selected_events>(),
    host_buffers.host_number_of_selected_events,
    sizeof(uint),
    cudaMemcpyHostToDevice, 
    cuda_stream));
  
  // Setup opts and arguments for kernel call
  state.set_opts(dim3(runtime_options.number_of_events), dim3(32), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_ut_raw_input>(),
    arguments.offset<dev_ut_raw_input_offsets>(),
    arguments.offset<dev_scifi_raw_input>(),
    arguments.offset<dev_scifi_raw_input_offsets>(),
    arguments.offset<dev_number_of_selected_events>(),
    arguments.offset<dev_event_list>() );
 
  state.invoke();

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_number_of_selected_events,
    arguments.offset<dev_number_of_selected_events>(),
    sizeof(uint),
    cudaMemcpyDeviceToHost, 
    cuda_stream));

  // TODO: This is not needed here
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_event_list,
    arguments.offset<dev_event_list>(),
    runtime_options.number_of_events*sizeof(uint),
    cudaMemcpyDeviceToHost, 
    cuda_stream));
    
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
  
  if (logger::ll.verbosityLevel >= logger::debug) {
    debug_cout << "Selected " << host_buffers.host_number_of_selected_events[0] << " / " << runtime_options.number_of_events << " events with global event cuts" << std::endl;
  }
}
