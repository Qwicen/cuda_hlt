#include "Logger.h"
#include "VeloPVIP.cuh"
#include "SequenceVisitor.cuh"

template<>
void SequenceVisitor::set_arguments_size<velo_pv_ip_t>(
  velo_pv_ip_t::arguments_t arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers& host_buffers)
{
  auto n_velo_tracks = host_buffers.host_number_of_reconstructed_velo_tracks[0];
  arguments.set_size<dev_velo_pv_ip>(Associate::Consolidated::Table::size(n_velo_tracks));
}

template<>
void SequenceVisitor::visit<velo_pv_ip_t>(
  velo_pv_ip_t& state,
  const velo_pv_ip_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts and arguments for kernel call
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(32), cuda_stream);
  state.set_arguments(arguments.offset<dev_velo_kalman_beamline_states>(),
                      arguments.offset<dev_atomics_velo>(),
                      arguments.offset<dev_velo_track_hit_number>(),
                      arguments.offset<dev_multi_fit_vertices>(),
                      arguments.offset<dev_number_of_multi_fit_vertices>(),
                      arguments.offset<dev_velo_pv_ip>());

  state.invoke();
}
