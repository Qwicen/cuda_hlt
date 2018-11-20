#include "SequenceVisitor.cuh"
#include "RunBeamlinePVonCPU.h"
#include "Tools.h"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(cpu_beamlinePV_t)

template<>
void SequenceVisitor::visit<cpu_beamlinePV_t>(
  cpu_beamlinePV_t& state,
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
    arguments.offset<dev_kalmanvelo_states>(),
    arguments.size<dev_kalmanvelo_states>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  state.invoke(
    host_buffers.host_kalmanvelo_states,
    host_buffers.host_velo_tracks_atomics,
    host_buffers.host_velo_track_hit_number,
    runtime_options.number_of_events);
}
