#include "SequenceVisitor.cuh"
#include "SearchByTriplet.cuh"

template<>
void SequenceVisitor::visit<search_by_triplet_t>(
  search_by_triplet_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_tracks>(runtime_options.number_of_events * VeloTracking::max_tracks);
  arguments.set_size<arg::dev_tracklets>(runtime_options.number_of_events * VeloTracking::ttf_modulo);
  arguments.set_size<arg::dev_tracks_to_follow>(runtime_options.number_of_events * VeloTracking::ttf_modulo);
  arguments.set_size<arg::dev_weak_tracks>(runtime_options.number_of_events * VeloTracking::max_weak_tracks);
  arguments.set_size<arg::dev_hit_used>(host_buffers.host_total_number_of_velo_clusters[0]);
  arguments.set_size<arg::dev_atomics_storage>(runtime_options.number_of_events * VeloTracking::num_atomics);
  arguments.set_size<arg::dev_rel_indices>(runtime_options.number_of_events * 2 * VeloTracking::max_numhits_in_module);
  scheduler.setup_next(arguments, sequence_step);

  // Setup opts and arguments
  state.set_opts(dim3(runtime_options.number_of_events), dim3(32), cuda_stream, 32 * sizeof(float));
  state.set_arguments(
    arguments.offset<arg::dev_velo_cluster_container>(),
    arguments.offset<arg::dev_estimated_input_size>(),
    arguments.offset<arg::dev_module_cluster_num>(),
    arguments.offset<arg::dev_tracks>(),
    arguments.offset<arg::dev_tracklets>(),
    arguments.offset<arg::dev_tracks_to_follow>(),
    arguments.offset<arg::dev_weak_tracks>(),
    arguments.offset<arg::dev_hit_used>(),
    arguments.offset<arg::dev_atomics_storage>(),
    arguments.offset<arg::dev_h0_candidates>(),
    arguments.offset<arg::dev_h2_candidates>(),
    arguments.offset<arg::dev_rel_indices>(),
    constants.dev_velo_module_zs
  );

  state.invoke();
}
