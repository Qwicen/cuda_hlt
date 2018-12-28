#include "SequenceVisitor.cuh"
#include "SearchWindows.cuh"

template<>
void SequenceVisitor::set_arguments_size<ut_search_windows_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_windows_layers>(6 * UT::Constants::n_layers * host_buffers.host_number_of_reconstructed_velo_tracks[0]);
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
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(64, UT::Constants::n_layers), cuda_stream);

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
    arguments.offset<dev_windows_layers>()
  );

  state.invoke();
}
