#include "SequenceVisitor.cuh"
#include "PrefixSumHandler.cuh"

template<>
void SequenceVisitor::set_arguments_size<prefix_sum_ut_hits_t>(
  prefix_sum_ut_hits_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_prefix_sum_auxiliary_array_3>(
    prefix_sum_ut_hits_t::aux_array_size(host_buffers.host_number_of_selected_events[0] * constants.host_unique_x_sector_layer_offsets[4]));
}

template<>
void SequenceVisitor::visit<prefix_sum_ut_hits_t>(
  prefix_sum_ut_hits_t& state,
  const prefix_sum_ut_hits_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Set size of the main array to be prefix summed
  state.set_size(host_buffers.host_number_of_selected_events[0] * constants.host_unique_x_sector_layer_offsets[4]);

  // Set the cuda_stream
  state.set_opts(cuda_stream);

  // Set arguments: Array to prefix sum and auxiliary array
  state.set_arguments(
    arguments.offset<dev_ut_hit_offsets>(),
    arguments.offset<dev_prefix_sum_auxiliary_array_3>()
  );

  // Invoke all steps of prefix sum
  state.invoke();

  // Fetch total number of hits accumulated with all tracks
  cudaCheck(cudaMemcpyAsync(host_buffers.host_accumulated_number_of_ut_hits,
    arguments.offset<dev_ut_hit_offsets>() + host_buffers.host_number_of_selected_events[0] * constants.host_unique_x_sector_layer_offsets[4],
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // // Now, we should have the offset instead, and the sum of all in host_accumulated_number_of_ut_hits
  // // Check that
  // cudaCheck(cudaMemcpyAsync(host_ut_hit_count.data(), arguments.offset<dev_ut_hit_offsets>(), arguments.size<dev_ut_hit_offsets>(), cudaMemcpyDeviceToHost, stream));
  // cudaEventRecord(cuda_generic_event, stream);
  // cudaEventSynchronize(cuda_generic_event);
  // for (int e=0; e<number_of_events; ++e) {
  //   info_cout << "Event " << e << ", offset per sector group: ";
  //   uint32_t* offset = host_ut_hit_count.data() + e * constants.host_unique_x_sector_layer_offsets[4];
  //   for (uint32_t i = 0; i < constants.host_unique_x_sector_layer_offsets[4]; ++i) {
  //     info_cout << offset[i] << ", ";
  //   }
  //   info_cout << std::endl;
  // }
  // info_cout << "Total number of UT hits: " << *host_accumulated_number_of_ut_hits << std::endl;
}
