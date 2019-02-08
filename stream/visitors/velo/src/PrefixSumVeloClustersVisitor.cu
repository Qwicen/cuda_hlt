#include "SequenceVisitor.cuh"
#include "PrefixSumHandler.cuh"

template<>
void SequenceVisitor::set_arguments_size<prefix_sum_velo_clusters_t>(
  prefix_sum_velo_clusters_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_cluster_offset>(prefix_sum_velo_clusters_t::aux_array_size(
    host_buffers.host_number_of_selected_events[0] * Velo::Constants::n_modules));
}

template<>
void SequenceVisitor::visit<prefix_sum_velo_clusters_t>(
  prefix_sum_velo_clusters_t& state,
  const prefix_sum_velo_clusters_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Set size of the main array to be prefix summed
  state.set_size(host_buffers.host_number_of_selected_events[0] * Velo::Constants::n_modules);

  // Set the cuda_stream
  state.set_opts(cuda_stream);

  // Set arguments: Array to prefix sum and auxiliary array
  state.set_arguments(arguments.offset<dev_estimated_input_size>(), arguments.offset<dev_cluster_offset>());

  // Invoke all steps of prefix sum
  state.invoke();

  // Fetch the number of hits we require
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_total_number_of_velo_clusters,
    arguments.offset<dev_estimated_input_size>() +
      host_buffers.host_number_of_selected_events[0] * Velo::Constants::n_modules,
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  if (logger::ll.verbosityLevel >= logger::debug) {
    debug_cout << "velo clusters = " << *host_buffers.host_total_number_of_velo_clusters << std::endl;
  }
}
