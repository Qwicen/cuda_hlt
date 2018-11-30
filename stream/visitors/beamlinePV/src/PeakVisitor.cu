#include "SequenceVisitor.cuh"
#include "blpv_peak.cuh"

template<>
void SequenceVisitor::set_arguments_size<blpv_peak_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  // Set arguments size
  arguments.set_size<dev_zpeaks>(runtime_options.number_of_events * PV::max_number_vertices);
  arguments.set_size<dev_number_of_zpeaks>(runtime_options.number_of_events);
}


template<>
void SequenceVisitor::visit<blpv_peak_t>(
  blpv_peak_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{

  state.set_opts(dim3(runtime_options.number_of_events), 1, cuda_stream);
  state.set_arguments(
    arguments.offset<dev_zhisto>(),
    arguments.offset<dev_zpeaks>(),
    arguments.offset<dev_number_of_zpeaks>()
  );


  state.invoke();

  //debugging
  /*

    // Retrieve result
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_peaks,
    arguments.offset<dev_zpeaks>(),
    arguments.size<dev_zpeaks>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));


    // Retrieve result
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_number_of_peaks,
    arguments.offset<dev_number_of_zpeaks>(),
    arguments.size<dev_number_of_zpeaks>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));

  // Wait to receive the result
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // Check the output
  for(int i_event = 0; i_event < runtime_options.number_of_events; i_event++) {
    std::cout << "event " << i_event << std::endl;
    for(int i = 0; i < host_buffers.host_number_of_peaks[i_event]; i++) {
      std::cout << "peak " << i << " " << host_buffers.host_peaks[i] << std::endl;
    }
  }
  */
  
  



    
}
