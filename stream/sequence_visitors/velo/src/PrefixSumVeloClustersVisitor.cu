#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

template<>
void SequenceVisitor::visit<prefix_sum_reduce_velo_clusters_t>(
  prefix_sum_reduce_velo_clusters_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Convert the estimated sizes to module hit start format (argument_offsets)
  // Set arguments and reserve memory
  arguments.set_size<arg::dev_cluster_offset>(runtime_options.number_of_events);
  scheduler.setup_next(arguments, sequence_step);
  
  // Setup sequence step
  const auto prefix_sum_blocks = (VeloTracking::n_modules * runtime_options.number_of_events + 511) / 512;
  state.set_opts(dim3(prefix_sum_blocks), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_estimated_input_size>(),
    arguments.offset<arg::dev_cluster_offset>(),
    VeloTracking::n_modules * runtime_options.number_of_events
  );

  // Kernel call
  state.invoke();
}

template<>
void SequenceVisitor::visit<prefix_sum_single_block_velo_clusters_t>(
  prefix_sum_single_block_velo_clusters_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // TODO: Make prefix sum use less repeated code
  const auto prefix_sum_blocks = (VeloTracking::n_modules * runtime_options.number_of_events + 511) / 512;

  // Prefix Sum Single Block
  scheduler.setup_next(arguments, sequence_step);
  state.set_opts(dim3(1), dim3(1024), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_estimated_input_size>() + VeloTracking::n_modules * runtime_options.number_of_events,
    arguments.offset<arg::dev_cluster_offset>(),
    prefix_sum_blocks
  );

  state.invoke();
}

template<>
void SequenceVisitor::visit<prefix_sum_scan_velo_clusters_t>(
  prefix_sum_scan_velo_clusters_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Prefix sum scan
  scheduler.setup_next(arguments, sequence_step);

  const auto prefix_sum_blocks = (VeloTracking::n_modules * runtime_options.number_of_events + 511) / 512;
  const auto prefix_sum_scan_blocks = prefix_sum_blocks==1 ? 1 : (prefix_sum_blocks-1);
  state.set_opts(dim3(prefix_sum_scan_blocks), dim3(512), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_estimated_input_size>(),
    arguments.offset<arg::dev_cluster_offset>(),
    VeloTracking::n_modules * runtime_options.number_of_events
  );
  state.invoke();

  // Fetch the number of hits we require
  cudaCheck(cudaMemcpyAsync(host_buffers.host_total_number_of_velo_clusters,
    arguments.offset<arg::dev_estimated_input_size>() + runtime_options.number_of_events * VeloTracking::n_modules,
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
}
