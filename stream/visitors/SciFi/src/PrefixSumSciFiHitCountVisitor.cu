#include "PrefixSumHandler.cuh"
#include "SequenceVisitor.cuh"

template<>
void SequenceVisitor::set_arguments_size<prefix_sum_scifi_hits_t>(
  prefix_sum_scifi_hits_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_prefix_sum_auxiliary_array_4>(prefix_sum_scifi_hits_t::aux_array_size(
    host_buffers.host_number_of_selected_events[0] * SciFi::Constants::n_mat_groups_and_mats));
}

template<>
void SequenceVisitor::visit<prefix_sum_scifi_hits_t>(
  prefix_sum_scifi_hits_t& state,
  const prefix_sum_scifi_hits_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Set size of the main array to be prefix summed
  state.set_size(host_buffers.host_number_of_selected_events[0] * SciFi::Constants::n_mat_groups_and_mats);

  // Set the cuda_stream
  state.set_opts(cuda_stream);

  // Set arguments: Array to prefix sum and auxiliary array
  state.set_arguments(arguments.offset<dev_scifi_hit_count>(), arguments.offset<dev_prefix_sum_auxiliary_array_4>());

  // Invoke all steps of prefix sum
  state.invoke();

  // Fetch total number of hits
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_accumulated_number_of_scifi_hits,
    arguments.offset<dev_scifi_hit_count>() +
      host_buffers.host_number_of_selected_events[0] * SciFi::Constants::n_mat_groups_and_mats,
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // info_cout << "Total SciFi cluster count: " << *host_buffers.host_accumulated_number_of_scifi_hits << std::endl;
}
