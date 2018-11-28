#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

template<>
void SequenceVisitor::set_arguments_size<prefix_sum_reduce_ut_track_hit_number_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  const size_t prefix_sum_auxiliary_array_size =
    (host_buffers.host_number_of_reconstructed_ut_tracks[0] + 511) / 512;
  arguments.set_size<dev_prefix_sum_auxiliary_array_5>(prefix_sum_auxiliary_array_size);
}

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(prefix_sum_single_block_ut_track_hit_number_t)
DEFINE_EMPTY_SET_ARGUMENTS_SIZE(prefix_sum_scan_ut_track_hit_number_t)

template<>
void SequenceVisitor::visit<prefix_sum_reduce_ut_track_hit_number_t>(
  prefix_sum_reduce_ut_track_hit_number_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Prefix sum: Reduce
  const size_t prefix_sum_auxiliary_array_size =
    (host_buffers.host_number_of_reconstructed_ut_tracks[0] + 511) / 512;

  state.set_opts(dim3(prefix_sum_auxiliary_array_size), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_ut_track_hit_number>(),
    arguments.offset<dev_prefix_sum_auxiliary_array_5>(),
    host_buffers.host_number_of_reconstructed_ut_tracks[0]
  );

  state.invoke();
}

template<>
void SequenceVisitor::visit<prefix_sum_single_block_ut_track_hit_number_t>(
  prefix_sum_single_block_ut_track_hit_number_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  const size_t prefix_sum_auxiliary_array_size =
    (host_buffers.host_number_of_reconstructed_ut_tracks[0] + 511) / 512;

  // Prefix sum: Single block
  state.set_opts(dim3(1), dim3(1024), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_ut_track_hit_number>() + host_buffers.host_number_of_reconstructed_ut_tracks[0],
    arguments.offset<dev_prefix_sum_auxiliary_array_5>(),
    prefix_sum_auxiliary_array_size
  );
  state.invoke();
}

template<>
void SequenceVisitor::visit<prefix_sum_scan_ut_track_hit_number_t>(
  prefix_sum_scan_ut_track_hit_number_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Prefix sum: Scan
  const size_t prefix_sum_auxiliary_array_size =
    (host_buffers.host_number_of_reconstructed_ut_tracks[0] + 511) / 512;
  const uint pss_ut_track_hit_number_opts =
    prefix_sum_auxiliary_array_size==1 ? 1 : (prefix_sum_auxiliary_array_size-1);

  state.set_opts(dim3(pss_ut_track_hit_number_opts), dim3(512), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_ut_track_hit_number>(),
    arguments.offset<dev_prefix_sum_auxiliary_array_5>(),
    host_buffers.host_number_of_reconstructed_ut_tracks[0]
  );

  state.invoke();

  // Fetch total number of hits accumulated with all tracks
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_accumulated_number_of_hits_in_ut_tracks,
    arguments.offset<dev_ut_track_hit_number>() + host_buffers.host_number_of_reconstructed_ut_tracks[0],
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
}
