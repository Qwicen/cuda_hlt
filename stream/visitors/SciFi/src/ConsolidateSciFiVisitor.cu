#include "SequenceVisitor.cuh"
#include "ConsolidateSciFi.cuh"

template<>
void SequenceVisitor::set_arguments_size<consolidate_scifi_tracks_t>(
  consolidate_scifi_tracks_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_scifi_track_hits>(host_buffers.host_accumulated_number_of_scifi_hits[0]*sizeof(SciFi::BaseHits));
  arguments.set_size<dev_scifi_qop>(host_buffers.host_number_of_reconstructed_scifi_tracks[0]);
  arguments.set_size<dev_scifi_track_ut_indices>(host_buffers.host_number_of_reconstructed_scifi_tracks[0]); 
  arguments.set_size<dev_scifi_states>(host_buffers.host_number_of_reconstructed_scifi_tracks[0]); 
}

template<>
void SequenceVisitor::visit<consolidate_scifi_tracks_t>(
  consolidate_scifi_tracks_t& state,
  const consolidate_scifi_tracks_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_scifi_hits>(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.offset<dev_scifi_track_hits>(),
    arguments.offset<dev_atomics_scifi>(),
    arguments.offset<dev_scifi_track_hit_number>(),
    arguments.offset<dev_scifi_qop>(),
    arguments.offset<dev_scifi_states>(),
    arguments.offset<dev_scifi_track_ut_indices>(),
    arguments.offset<dev_scifi_tracks>(),
    constants.dev_scifi_geometry,
    constants.dev_inv_clus_res
  );

  state.invoke();

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // Transmission device to host of Scifi consolidated tracks
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_atomics_scifi,
    arguments.offset<dev_atomics_scifi>(),
    (2 * host_buffers.host_number_of_selected_events[0] + 1) * sizeof(int),
    cudaMemcpyDeviceToHost, 
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_scifi_track_hit_number,
    arguments.offset<dev_scifi_track_hit_number>(),
    arguments.size<dev_scifi_track_hit_number>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_scifi_track_hits,
    arguments.offset<dev_scifi_track_hits>(),
    host_buffers.host_accumulated_number_of_hits_in_scifi_tracks[0]*sizeof(SciFi::Hit), 
    cudaMemcpyDeviceToHost,
    cuda_stream));
  
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_scifi_qop,
    arguments.offset<dev_scifi_qop>(),
    arguments.size<dev_scifi_qop>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_scifi_track_ut_indices,
    arguments.offset<dev_scifi_track_ut_indices>(),
    arguments.size<dev_scifi_track_ut_indices>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));
}
