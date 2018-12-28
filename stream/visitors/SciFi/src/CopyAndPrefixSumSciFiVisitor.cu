#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(copy_and_prefix_sum_single_block_scifi_t)

template<>
void SequenceVisitor::visit<copy_and_prefix_sum_single_block_scifi_t>(
  copy_and_prefix_sum_single_block_scifi_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(1), dim3(1024), cuda_stream);
  state.set_arguments(
    (uint*) arguments.offset<dev_atomics_scifi>() + host_buffers.host_number_of_selected_events[0]*2,
    (uint*) arguments.offset<dev_atomics_scifi>(),
    (uint*) arguments.offset<dev_atomics_scifi>() + host_buffers.host_number_of_selected_events[0],
    host_buffers.host_number_of_selected_events[0]
  );
  
  state.invoke();

  // Fetch number of reconstructed SciFi tracks.
  cudaCheck(
    cudaMemcpyAsync(
      host_buffers.host_number_of_reconstructed_scifi_tracks,
      arguments.offset<dev_atomics_scifi>() + host_buffers.host_number_of_selected_events[0]*2,
      sizeof(uint),
      cudaMemcpyDeviceToHost,
      cuda_stream
    )
  );

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  debug_cout << "Total # of SciFi tracks = " << host_buffers.host_number_of_reconstructed_scifi_tracks[0]  << std::endl;
}
