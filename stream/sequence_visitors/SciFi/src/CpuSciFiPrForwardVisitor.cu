#include "SequenceVisitor.cuh"
#include "RunForwardCPU.h"
#include "Tools.h"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(cpu_scifi_pr_forward_t)

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
  std::vector<uint> host_scifi_hit_count (2 * runtime_options.number_of_events * SciFi::Constants::n_zones + 1);

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
    host_buffers.forward_tracks_events,
    host_scifi_hits.data(),
    host_scifi_hit_count.data(),
    constants.host_scifi_geometry,
    host_buffers.host_velo_tracks_atomics,
    host_buffers.host_velo_track_hit_number,
    (uint*) host_buffers.host_velo_states,
    host_buffers.host_veloUT_tracks,
    host_buffers.host_atomics_veloUT,
    runtime_options.number_of_events,
    constants.host_inv_clus_res);
}
