#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"

template<>
void SequenceVisitor::visit<prefix_sum_reduce_velo_track_hit_number_t>(
  prefix_sum_reduce_velo_track_hit_number_t& state,
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
  const size_t prefix_sum_auxiliary_array_size =
    (host_buffers.host_number_of_reconstructed_velo_tracks[0] + 511) / 512;
  arguments.set_size<arg::dev_prefix_sum_auxiliary_array_2>(prefix_sum_auxiliary_array_size);
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(prefix_sum_auxiliary_array_size), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_velo_track_hit_number>(),
    arguments.offset<arg::dev_prefix_sum_auxiliary_array_2>(),
    host_buffers.host_number_of_reconstructed_velo_tracks[0]
  );

  state.invoke();
}

template<>
void SequenceVisitor::visit<prefix_sum_single_block_velo_track_hit_number_t>(
  prefix_sum_single_block_velo_track_hit_number_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  const size_t prefix_sum_auxiliary_array_size =
    (host_buffers.host_number_of_reconstructed_velo_tracks[0] + 511) / 512;

  // Prefix sum: Single block
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(1), dim3(1024), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_velo_track_hit_number>() + host_buffers.host_number_of_reconstructed_velo_tracks[0],
    arguments.offset<arg::dev_prefix_sum_auxiliary_array_2>(),
    prefix_sum_auxiliary_array_size
  );
  state.invoke();
}

template<>
void SequenceVisitor::visit<prefix_sum_scan_velo_track_hit_number_t>(
  prefix_sum_scan_velo_track_hit_number_t& state,
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
  const size_t prefix_sum_auxiliary_array_size =
    (host_buffers.host_number_of_reconstructed_velo_tracks[0] + 511) / 512;
  const uint pss_velo_track_hit_number_opts =
    prefix_sum_auxiliary_array_size==1 ? 1 : (prefix_sum_auxiliary_array_size-1);

  state.set_opts(dim3(pss_velo_track_hit_number_opts), dim3(512), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_velo_track_hit_number>(),
    arguments.offset<arg::dev_prefix_sum_auxiliary_array_2>(),
    host_buffers.host_number_of_reconstructed_velo_tracks[0]
  );

  state.invoke();

  // Fetch total number of hits accumulated with all tracks
  cudaCheck(cudaMemcpyAsync(host_buffers.host_accumulated_number_of_hits_in_velo_tracks,
    arguments.offset<arg::dev_velo_track_hit_number>() + host_buffers.host_number_of_reconstructed_velo_tracks[0],
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
}
