#include "SequenceVisitor.cuh"
#include "VeloUT.cuh"

template<>
void SequenceVisitor::set_arguments_size<veloUT_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_veloUT_tracks>(runtime_options.number_of_events * VeloUTTracking::max_num_tracks);
  arguments.set_size<dev_atomics_veloUT>(runtime_options.number_of_events * VeloUTTracking::num_atomics);
}

template<>
void SequenceVisitor::visit<veloUT_t>(
  veloUT_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(runtime_options.number_of_events), dim3(32), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_ut_hits>(),
    arguments.offset<dev_ut_hit_offsets>(),
    arguments.offset<dev_atomics_storage>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_velo_track_hits>(),
    arguments.offset<dev_velo_states>(),
    arguments.offset<dev_veloUT_tracks>(),
    arguments.offset<dev_atomics_veloUT>(),
    constants.dev_ut_magnet_tool,
    constants.dev_ut_dxDy,
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_x_sector_offsets,
    constants.dev_unique_sector_xs
  );

  state.invoke();

  // TODO: Maybe this should not go here
  // Fetch all UT tracks
  cudaCheck(cudaMemcpyAsync(host_buffers.host_atomics_veloUT,
    arguments.offset<dev_atomics_veloUT>(),
    arguments.size<dev_atomics_veloUT>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(host_buffers.host_veloUT_tracks,
    arguments.offset<dev_veloUT_tracks>(),
    arguments.size<dev_veloUT_tracks>(),
    cudaMemcpyDeviceToHost, 
    cuda_stream));
}
