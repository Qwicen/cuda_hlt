#include "SequenceVisitor.cuh"
#include "PrForward.cuh"

template<>
void SequenceVisitor::set_arguments_size<scifi_pr_forward_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_scifi_tracks>(runtime_options.number_of_events * SciFi::max_tracks);
  arguments.set_size<dev_n_scifi_tracks>(runtime_options.number_of_events);
}

template<>
void SequenceVisitor::visit<scifi_pr_forward_t>(
  scifi_pr_forward_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(runtime_options.number_of_events), dim3(32), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_scifi_hits>(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.offset<dev_atomics_storage>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_velo_states>(),
    arguments.offset<dev_compassUT_tracks>(),
    arguments.offset<dev_atomics_compassUT>(),
    arguments.offset<dev_scifi_tracks>(),
    arguments.offset<dev_n_scifi_tracks>(),
    constants.dev_scifi_tmva1,
    constants.dev_scifi_tmva2,
    constants.dev_scifi_constArrays
  );
  state.invoke(); 
  
  // Transmission device to host
  // SciFi tracks
  cudaCheck(cudaMemcpyAsync(host_buffers.host_n_scifi_tracks,
    arguments.offset<dev_n_scifi_tracks>(),
    arguments.size<dev_n_scifi_tracks>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(host_buffers.host_scifi_tracks,
    arguments.offset<dev_scifi_tracks>(),
    arguments.size<dev_scifi_tracks>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));
}
