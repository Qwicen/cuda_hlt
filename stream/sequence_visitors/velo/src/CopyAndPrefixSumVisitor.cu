#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

template<>
void SequenceVisitor::visit<copy_and_prefix_sum_single_block_t>(
  copy_and_prefix_sum_single_block_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Calculate prefix sum of found tracks
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(1), dim3(1024), cuda_stream);
  state.set_arguments(
    (uint*) arguments.offset<arg::dev_atomics_storage>() + runtime_options.number_of_events*2,
    (uint*) arguments.offset<arg::dev_atomics_storage>(),
    (uint*) arguments.offset<arg::dev_atomics_storage>() + runtime_options.number_of_events,
    runtime_options.number_of_events
  );

  state.invoke();

  // Fetch number of reconstructed tracks
  cudaCheck(cudaMemcpyAsync(host_buffers.host_number_of_reconstructed_velo_tracks,
    arguments.offset<arg::dev_atomics_storage>() + runtime_options.number_of_events * 2,
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
}
