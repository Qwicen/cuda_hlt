#include "SequenceVisitor.cuh"
#include "PrefixSumHandler.cuh"

template<>
void SequenceVisitor::set_arguments_size<prefix_sum_velo_track_hit_number_t>(
  prefix_sum_velo_track_hit_number_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_prefix_sum_auxiliary_array_2>(
    prefix_sum_velo_track_hit_number_t::aux_array_size(host_buffers.host_number_of_reconstructed_velo_tracks[0]));
}

template<>
void SequenceVisitor::visit<prefix_sum_velo_track_hit_number_t>(
  prefix_sum_velo_track_hit_number_t& state,
  const prefix_sum_velo_track_hit_number_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Set size of the main array to be prefix summed
  state.set_size(host_buffers.host_number_of_reconstructed_velo_tracks[0]);

  // Set the cuda_stream
  state.set_opts(cuda_stream);

  // Set arguments: Array to prefix sum and auxiliary array
  state.set_arguments(
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_prefix_sum_auxiliary_array_2>()
  );

  // Invoke all steps of prefix sum
  state.invoke();

  // Fetch total number of hits accumulated with all tracks
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_accumulated_number_of_hits_in_velo_tracks,
    arguments.offset<dev_velo_track_hit_number>() + host_buffers.host_number_of_reconstructed_velo_tracks[0],
    sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
  
  if (logger::ll.verbosityLevel >= logger::debug) {
    debug_cout << "Total number of hits on all tracks = " << host_buffers.host_accumulated_number_of_hits_in_velo_tracks[0] << std::endl;
  }
}
