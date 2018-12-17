#include "SequenceVisitor.cuh"
#include "CompassUT.cuh"

template<>
void SequenceVisitor::set_arguments_size<compass_ut_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_compassUT_tracks>(runtime_options.number_of_events * VeloUTTracking::max_num_tracks);
  arguments.set_size<dev_atomics_compassUT>(runtime_options.number_of_events * VeloUTTracking::num_atomics);
  // arguments.set_size<dev_active_tracks>(runtime_options.number_of_events);
}

template<>
void SequenceVisitor::visit<compass_ut_t>(
  compass_ut_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(runtime_options.number_of_events), dim3(VeloUTTracking::num_threads), cuda_stream);

  state.set_arguments(
    arguments.offset<dev_ut_hits>(),
    arguments.offset<dev_ut_hit_offsets>(),
    arguments.offset<dev_atomics_storage>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_velo_track_hits>(),
    arguments.offset<dev_velo_states>(),
    constants.dev_ut_magnet_tool,
    constants.dev_ut_dxDy,
    arguments.offset<dev_active_tracks>(),
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_sector_xs,
    arguments.offset<dev_compassUT_tracks>(),
    arguments.offset<dev_atomics_compassUT>(),
    arguments.offset<dev_windows_layers>()    
  );

  cudaCheck(cudaMemsetAsync(arguments.offset<dev_active_tracks>(), 0, arguments.size<dev_active_tracks>(), cuda_stream));
  cudaCheck(cudaMemsetAsync(arguments.offset<dev_atomics_compassUT>(), 0, arguments.size<dev_atomics_compassUT>(), cuda_stream));

  state.invoke();

  // TODO: Maybe this should not go here
  // Fetch all UT tracks
  cudaCheck(cudaMemcpyAsync(host_buffers.host_atomics_compassUT,
    arguments.offset<dev_atomics_compassUT>(),
    arguments.size<dev_atomics_compassUT>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(host_buffers.host_compassUT_tracks,
    arguments.offset<dev_compassUT_tracks>(),
    arguments.size<dev_compassUT_tracks>(),
    cudaMemcpyDeviceToHost, 
    cuda_stream));
}
