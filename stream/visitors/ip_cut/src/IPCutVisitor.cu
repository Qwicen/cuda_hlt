#include "Logger.h"
#include "IPCut.cuh"
#include "SequenceVisitor.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(ip_cut_t)

template<>
void SequenceVisitor::visit<ip_cut_t>(
  ip_cut_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts and arguments for kernel call
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(32), cuda_stream);
  state.set_arguments(arguments.offset<dev_velo_kalman_beamline_states>(),
                      arguments.offset<dev_atomics_velo>(),
                      arguments.offset<dev_velo_track_hit_number>(),
                      arguments.offset<dev_velo_pv_ip>(),
                      arguments.offset<dev_accepted_velo_tracks>());

  state.invoke();
}
