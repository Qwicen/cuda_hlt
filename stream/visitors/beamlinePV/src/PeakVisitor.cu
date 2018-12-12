#include "SequenceVisitor.cuh"
#include "pv_beamline_peak.cuh"

template<>
void SequenceVisitor::set_arguments_size<pv_beamline_peak_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  // Set arguments size
  arguments.set_size<dev_zpeaks>(host_buffers.host_number_of_selected_events[0] * PV::max_number_vertices);
  arguments.set_size<dev_number_of_zpeaks>(host_buffers.host_number_of_selected_events[0]);
}

template<>
void SequenceVisitor::visit<pv_beamline_peak_t>(
  pv_beamline_peak_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(
    dim3((host_buffers.host_number_of_selected_events[0] + PV::num_threads_pv_beamline_peak_t - 1) / PV::num_threads_pv_beamline_peak_t),
    PV::num_threads_pv_beamline_peak_t,
    cuda_stream);

  state.set_arguments(
    arguments.offset<dev_zhisto>(),
    arguments.offset<dev_zpeaks>(),
    arguments.offset<dev_number_of_zpeaks>(),
    host_buffers.host_number_of_selected_events[0]);

  state.invoke();

  // // Debugging

  // // Retrieve result
  // cudaCheck(cudaMemcpyAsync(
  //   host_buffers.host_peaks,
  //   arguments.offset<dev_zpeaks>(),
  //   arguments.size<dev_zpeaks>(),
  //   cudaMemcpyDeviceToHost,
  //   cuda_stream));

  // // Retrieve result
  // cudaCheck(cudaMemcpyAsync(
  //   host_buffers.host_number_of_peaks,
  //   arguments.offset<dev_number_of_zpeaks>(),
  //   arguments.size<dev_number_of_zpeaks>(),
  //   cudaMemcpyDeviceToHost,
  //   cuda_stream));

  // // Wait to receive the result
  // cudaEventRecord(cuda_generic_event, cuda_stream);
  // cudaEventSynchronize(cuda_generic_event);

  // // Check the output
  // for (int i_event = 0; i_event < host_buffers.host_number_of_selected_events[0]; i_event++) {
  //   std::cout << "event " << i_event << std::endl;
  //   for (int i = 0; i < host_buffers.host_number_of_peaks[i_event]; i++) {
  //     std::cout << "peak " << i << " " << host_buffers.host_peaks[i] << std::endl;
  //   }
  // }
}
