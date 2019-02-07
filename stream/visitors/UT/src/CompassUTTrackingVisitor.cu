#include "CompassUT.cuh"
#include "SequenceVisitor.cuh"

template<>
void SequenceVisitor::set_arguments_size<compass_ut_t>(
  compass_ut_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_ut_tracks>(host_buffers.host_number_of_selected_events[0] * UT::Constants::max_num_tracks);
  arguments.set_size<dev_atomics_ut>(host_buffers.host_number_of_selected_events[0] * UT::num_atomics + 1);
  // dev_ut_active_tracks init in search_windows kernel
}

template<>
void SequenceVisitor::visit<compass_ut_t>(
  compass_ut_t& state,
  const compass_ut_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(UT::Constants::num_thr_compassut), cuda_stream);

  state.set_arguments(
    arguments.offset<dev_ut_hits>(),
    arguments.offset<dev_ut_hit_offsets>(),
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_velo_track_hits>(),
    arguments.offset<dev_velo_states>(),
    constants.dev_ut_magnet_tool,
    constants.dev_ut_dxDy,
    arguments.offset<dev_ut_active_tracks>(),
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_sector_xs,
    arguments.offset<dev_ut_tracks>(),
    arguments.offset<dev_atomics_ut>(),
    arguments.offset<dev_ut_windows_layers>(),
    arguments.offset<dev_accepted_velo_tracks>()
  );

  cudaCheck(cudaMemsetAsync(arguments.offset<dev_ut_active_tracks>(), 0, arguments.size<dev_ut_active_tracks>(), cuda_stream));
  cudaCheck(cudaMemsetAsync(arguments.offset<dev_atomics_ut>(), 0, arguments.size<dev_atomics_ut>(), cuda_stream));

  state.invoke();
}
