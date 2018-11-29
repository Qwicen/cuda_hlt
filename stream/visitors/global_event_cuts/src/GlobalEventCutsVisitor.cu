#include "SequenceVisitor.cuh" 
#include "GlobalEventCuts.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(global_event_cuts_t)


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

  // Setup opts and arguments for kernel call
  state.set_opts(dim3(1), dim3(runtime_options.number_of_events), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_raw_input>(),
    arguments.offset<dev_raw_input_offsets>(),
    arguments.offset<dev_ut_raw_input>(),
    arguments.offset<dev_ut_raw_input_offsets>(),
    arguments.offset<dev_scifi_raw_input>(),
    arguments.offset<dev_scifi_raw_input_offsets>() );
 
 
}
