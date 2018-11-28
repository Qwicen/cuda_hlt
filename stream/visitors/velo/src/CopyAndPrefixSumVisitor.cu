#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(copy_and_prefix_sum_single_block_velo_t)

template<>
void SequenceVisitor::visit<copy_and_prefix_sum_single_block_velo_t>(
  copy_and_prefix_sum_single_block_velo_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(1), dim3(1024), cuda_stream);
  state.set_arguments(
    (uint*) arguments.offset<dev_atomics_velo>() + runtime_options.number_of_events*2,
    (uint*) arguments.offset<dev_atomics_velo>(),
    (uint*) arguments.offset<dev_atomics_velo>() + runtime_options.number_of_events,
    runtime_options.number_of_events
  );

  state.invoke();

  // Fetch number of reconstructed tracks
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_number_of_reconstructed_velo_tracks,
    arguments.offset<dev_atomics_velo>() + runtime_options.number_of_events * 2,
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
}