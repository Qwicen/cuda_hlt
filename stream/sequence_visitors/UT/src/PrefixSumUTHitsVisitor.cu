#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

template<>
void SequenceVisitor::visit<prefix_sum_reduce_ut_hits_t>(
  prefix_sum_reduce_ut_hits_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Prefix sum: Reduce
  const uint total_number_of_sectors = runtime_options.number_of_events * constants.host_unique_x_sector_layer_offsets[4];
  const size_t prefix_sum_auxiliary_array_size = (total_number_of_sectors + 511) / 512;
  arguments.set_size<arg::dev_prefix_sum_auxiliary_array_3>(prefix_sum_auxiliary_array_size);
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(prefix_sum_auxiliary_array_size), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_ut_hit_offsets>(),
    arguments.offset<arg::dev_prefix_sum_auxiliary_array_3>(),
    total_number_of_sectors
  );

  state.invoke();
}

template<>
void SequenceVisitor::visit<prefix_sum_single_block_ut_hits_t>(
  prefix_sum_single_block_ut_hits_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Prefix sum: Single block
  scheduler.setup_next(arguments, sequence_step);

  const uint total_number_of_sectors = runtime_options.number_of_events * constants.host_unique_x_sector_layer_offsets[4];
  const size_t prefix_sum_auxiliary_array_size = (total_number_of_sectors + 511) / 512;
  state.set_opts(dim3(1), dim3(1024), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_ut_hit_offsets>() + total_number_of_sectors,
    arguments.offset<arg::dev_prefix_sum_auxiliary_array_3>(),
    prefix_sum_auxiliary_array_size
  );

  state.invoke();
}

template<>
void SequenceVisitor::visit<prefix_sum_scan_ut_hits_t>(
  prefix_sum_scan_ut_hits_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Prefix sum: Scan
  scheduler.setup_next(arguments, sequence_step);

  const uint total_number_of_sectors = runtime_options.number_of_events * constants.host_unique_x_sector_layer_offsets[4];
  const size_t prefix_sum_auxiliary_array_size = (total_number_of_sectors + 511) / 512;
  const uint pss_ut_hits_blocks = prefix_sum_auxiliary_array_size==1 ? 1 : (prefix_sum_auxiliary_array_size-1);
  state.set_opts(dim3(pss_ut_hits_blocks), dim3(512), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_ut_hit_offsets>(),
    arguments.offset<arg::dev_prefix_sum_auxiliary_array_3>(),
    total_number_of_sectors
  );
  state.invoke();

  // Fetch total number of hits accumulated with all tracks
  cudaCheck(cudaMemcpyAsync(host_buffers.host_accumulated_number_of_ut_hits,
    arguments.offset<arg::dev_ut_hit_offsets>() + total_number_of_sectors,
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // // Now, we should have the offset instead, and the sum of all in host_accumulated_number_of_ut_hits
  // // Check that
  // cudaCheck(cudaMemcpyAsync(host_ut_hit_count.data(), arguments.offset<arg::dev_ut_hit_offsets>(), arguments.size<arg::dev_ut_hit_offsets>(), cudaMemcpyDeviceToHost, stream));
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
