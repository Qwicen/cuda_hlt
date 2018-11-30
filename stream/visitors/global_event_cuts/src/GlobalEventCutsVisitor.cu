#include "SequenceVisitor.cuh" 
#include "GlobalEventCuts.cuh"

template<>
void SequenceVisitor::set_arguments_size<global_event_cuts_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_number_of_selected_events>(1);
} 
 


template<>
void SequenceVisitor::visit<global_event_cuts_t>(
  global_event_cuts_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
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

  // cudaCheck(cudaMemcpyAsync(
  //   host_buffers.host_event_list,
  //   arguments.offset<dev_event_list>(),
  //   runtime_options.number_of_events*sizeof(uint),
  //   cudaMemcpyDeviceToHost, 
  //   cuda_stream));
    
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
  
  info_cout << "Selected " << host_buffers.host_number_of_selected_events[0] << " / " << runtime_options.number_of_events << " events with global event cuts" << std::endl;
  // for ( int i = 0; i < host_buffers.host_number_of_selected_events[0]; ++i ) {
  //   debug_cout << host_buffers.host_event_list[i] << std::endl;
  // }

}
