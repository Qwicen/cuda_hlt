#include "StreamVisitor.cuh"
#include "SearchWindows.cuh"

template<>
void StreamVisitor::visit<ut_search_windows_t>(
  ut_search_windows_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_windows_layers>(6 * VeloUTTracking::n_layers * host_buffers.host_number_of_reconstructed_velo_tracks[0]);
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events), dim3(64, VeloUTTracking::n_layers), cuda_stream);

  state.set_arguments(
    arguments.offset<arg::dev_ut_hits>(),
    arguments.offset<arg::dev_ut_hit_offsets>(),
    arguments.offset<arg::dev_atomics_storage>(),
    arguments.offset<arg::dev_velo_track_hit_number>(),
    arguments.offset<arg::dev_velo_states>(),
    constants.dev_ut_magnet_tool,
    constants.dev_ut_dxDy,
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_sector_xs,
    arguments.offset<arg::dev_windows_layers>()
  );

  state.invoke();
}