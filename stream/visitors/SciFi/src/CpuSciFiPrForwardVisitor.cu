#include "SequenceVisitor.cuh"
#include "RunForwardCPU.h"
#include "Tools.h"

template<> 
void SequenceVisitor::set_arguments_size<cpu_scifi_pr_forward_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_scifi_tracks>(runtime_options.number_of_events * SciFi::Constants::max_tracks);
  arguments.set_size<dev_atomics_scifi>(runtime_options.number_of_events * SciFi::num_atomics);
}

template<>
void SequenceVisitor::visit<cpu_scifi_pr_forward_t>(
  cpu_scifi_pr_forward_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Synchronize previous CUDA transmissions
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // Run Forward on x86 architecture
  std::vector<uint> host_scifi_hits (host_buffers.scifi_hits_uints());
  std::vector<uint> host_scifi_hit_count (2 * runtime_options.number_of_events * SciFi::Constants::n_mat_groups_and_mats + 1);
    
  host_buffers.scifi_tracks_events.reserve(runtime_options.number_of_events * SciFi::Constants::max_tracks);

  cudaCheck(cudaMemcpyAsync(
    host_scifi_hits.data(),
    arguments.offset<dev_scifi_hits>(),
    arguments.size<dev_scifi_hits>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(
    host_scifi_hit_count.data(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.size<dev_scifi_hit_count>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  // TODO: Maybe use this rv somewhere?
  int rv = state.invoke(
    host_buffers.scifi_tracks_events.data(),
    host_buffers.host_atomics_scifi,
    host_scifi_hits.data(),
    host_scifi_hit_count.data(),
    constants.host_scifi_geometry,
    constants.host_inv_clus_res, 
    host_buffers.host_atomics_velo,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_states,
    host_buffers.host_atomics_ut,
    host_buffers.host_ut_track_hit_number,
    host_buffers.host_ut_qop,
    host_buffers.host_ut_track_velo_indices,
    runtime_options.number_of_events);
 
  // copy SciFi track to device for consolidation
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_atomics_scifi>(), 
    host_buffers.host_atomics_scifi,
    arguments.size<dev_atomics_scifi>(),
    cudaMemcpyHostToDevice, 
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_scifi_tracks>(),
    host_buffers.scifi_tracks_events.data(),
    arguments.size<dev_scifi_tracks>(),
    cudaMemcpyHostToDevice,
    cuda_stream));
  
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
  
}
