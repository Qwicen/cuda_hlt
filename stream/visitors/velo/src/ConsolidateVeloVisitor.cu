#include "SequenceVisitor.cuh"
#include "ConsolidateVelo.cuh"
#include "States.cuh"

template<>
void SequenceVisitor::set_arguments_size<consolidate_velo_tracks_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_velo_track_hits>(host_buffers.host_accumulated_number_of_hits_in_velo_tracks[0] * sizeof(Velo::Hit));
  arguments.set_size<dev_velo_states>(host_buffers.host_number_of_reconstructed_velo_tracks[0] * sizeof(VeloState));
}

template<>
void SequenceVisitor::visit<consolidate_velo_tracks_t>(
  consolidate_velo_tracks_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_tracks>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_velo_cluster_container>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_module_cluster_num>(),
    arguments.offset<dev_velo_track_hits>(),
    arguments.offset<dev_velo_states>()
  );

  state.invoke();

  // TODO: Perhaps this shouldn't go here

  // Transmission device to host
  // Velo tracks
  cudaCheck(cudaMemcpyAsync(host_buffers.host_atomics_velo,
    arguments.offset<dev_atomics_velo>(),
    (2 * host_buffers.host_number_of_selected_events[0] + 1) * sizeof(uint),
    cudaMemcpyDeviceToHost, 
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(host_buffers.host_velo_track_hit_number,
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.size<dev_velo_track_hit_number>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(host_buffers.host_velo_track_hits,
    arguments.offset<dev_velo_track_hits>(),
                            host_buffers.host_accumulated_number_of_hits_in_velo_tracks[0] * sizeof(Velo::Hit), 
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(host_buffers.host_velo_states,
    arguments.offset<dev_velo_states>(),
    host_buffers.host_number_of_reconstructed_velo_tracks[0] * sizeof(VeloState),
    cudaMemcpyDeviceToHost,
    cuda_stream));
}
