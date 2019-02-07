#include "SequenceVisitor.cuh"
#include "GetSeeds.cuh"

template<>
void SequenceVisitor::set_arguments_size<pv_get_seeds_t>(
  pv_get_seeds_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  // Set arguments size
  arguments.set_size<dev_seeds>(host_buffers.host_number_of_reconstructed_velo_tracks[0] );
  arguments.set_size<dev_number_seeds>(host_buffers.host_number_of_selected_events[0] );
}



template<>
void SequenceVisitor::visit<pv_get_seeds_t>(
  pv_get_seeds_t& state,
  const pv_get_seeds_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{

  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), 1, cuda_stream);
  state.set_arguments(
    arguments.offset<dev_velo_kalman_beamline_states>(),
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_seeds>(),
    arguments.offset<dev_number_seeds>()
  );

  state.invoke();

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_number_of_seeds,
    arguments.offset<dev_number_seeds>(),
    arguments.size<dev_number_seeds>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));





    
}
