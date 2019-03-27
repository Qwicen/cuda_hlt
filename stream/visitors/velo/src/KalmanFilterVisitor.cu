#include "SequenceVisitor.cuh"
#include "VeloKalmanFilter.cuh"

template<>
void SequenceVisitor::set_arguments_size<velo_kalman_fit_t>(
  velo_kalman_fit_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  // Set arguments size
  arguments.set_size<dev_velo_kalman_beamline_states>(
    host_buffers.host_number_of_reconstructed_velo_tracks[0] * sizeof(VeloState));
}

template<>
void SequenceVisitor::visit<velo_kalman_fit_t>(
  velo_kalman_fit_t& state,
  const velo_kalman_fit_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(256), cuda_stream);

  state.set_arguments(
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_velo_track_hits>(),
    arguments.offset<dev_velo_states>(),
    arguments.offset<dev_velo_kalman_beamline_states>());

  state.invoke();

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_kalmanvelo_states,
    arguments.offset<dev_velo_kalman_beamline_states>(),
    arguments.size<dev_velo_kalman_beamline_states>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));
}
