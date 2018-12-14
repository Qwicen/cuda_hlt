#include "SequenceVisitor.cuh"
#include "RunBeamlinePVOnCPU.h"
#include "Tools.h"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(cpu_pv_beamline_t)

template<>
void SequenceVisitor::visit<cpu_pv_beamline_t>(
  cpu_pv_beamline_t& state,
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

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_kalmanvelo_states,
    arguments.offset<dev_velo_kalman_beamline_states>(),
    arguments.size<dev_velo_kalman_beamline_states>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  state.invoke(
    host_buffers.host_kalmanvelo_states,
    host_buffers.host_atomics_velo,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_reconstructed_pvs,
    host_buffers.host_number_of_vertex,
    host_buffers.host_number_of_selected_events[0]);
  
  for ( int i_event = 0; i_event < host_buffers.host_number_of_selected_events[0]; i_event++ ) {
    debug_cout << "# of PVs found = " << host_buffers.host_number_of_vertex[i_event] << std::endl;
  }

}
