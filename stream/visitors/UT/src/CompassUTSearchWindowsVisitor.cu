#include "SearchWindows.cuh"
#include "SequenceVisitor.cuh"

template<>
void SequenceVisitor::set_arguments_size<ut_search_windows_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_ut_windows_layers>(CompassUT::num_elems * UT::Constants::n_layers * host_buffers.host_number_of_reconstructed_velo_tracks[0]);
  arguments.set_size<dev_ut_active_tracks>(runtime_options.number_of_events);
}

template<>
void SequenceVisitor::visit<ut_search_windows_t>(
  ut_search_windows_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(UT::Constants::n_layers, UT::Constants::num_thr_searchwin), cuda_stream);

  state.set_arguments(
    arguments.offset<dev_ut_hits>(),
    arguments.offset<dev_ut_hit_offsets>(),
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_velo_states>(),
    constants.dev_ut_magnet_tool,
    constants.dev_ut_dxDy,
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_sector_xs,
    arguments.offset<dev_ut_windows_layers>(),
    arguments.offset<dev_ut_active_tracks>(),
    arguments.offset<dev_accepted_velo_tracks>()
  );

  cudaCheck(cudaMemsetAsync(arguments.offset<dev_ut_active_tracks>(), 0, arguments.size<dev_ut_active_tracks>(), cuda_stream));

  state.invoke();
}
