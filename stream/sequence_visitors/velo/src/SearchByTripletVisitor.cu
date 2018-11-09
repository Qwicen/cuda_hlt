#include "SequenceVisitor.cuh"
#include "SearchByTriplet.cuh"

template<>
void SequenceVisitor::set_arguments_size<search_by_triplet_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_tracks>(runtime_options.number_of_events * VeloTracking::max_tracks);
  arguments.set_size<dev_tracklets>(runtime_options.number_of_events * VeloTracking::ttf_modulo);
  arguments.set_size<dev_tracks_to_follow>(runtime_options.number_of_events * VeloTracking::ttf_modulo);
  arguments.set_size<dev_weak_tracks>(runtime_options.number_of_events * VeloTracking::max_weak_tracks);
  arguments.set_size<dev_hit_used>(host_buffers.host_total_number_of_velo_clusters[0]);
  arguments.set_size<dev_atomics_storage>(runtime_options.number_of_events * VeloTracking::num_atomics);
  arguments.set_size<dev_rel_indices>(runtime_options.number_of_events * 2 * VeloTracking::max_numhits_in_module);
}

template<>
void SequenceVisitor::visit<search_by_triplet_t>(
  search_by_triplet_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts and arguments
  state.set_opts(dim3(runtime_options.number_of_events), dim3(32), cuda_stream, 32 * sizeof(float));
  state.set_arguments(
    arguments.offset<dev_velo_cluster_container>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_module_cluster_num>(),
    arguments.offset<dev_tracks>(),
    arguments.offset<dev_tracklets>(),
    arguments.offset<dev_tracks_to_follow>(),
    arguments.offset<dev_weak_tracks>(),
    arguments.offset<dev_hit_used>(),
    arguments.offset<dev_atomics_storage>(),
    arguments.offset<dev_h0_candidates>(),
    arguments.offset<dev_h2_candidates>(),
    arguments.offset<dev_rel_indices>(),
    constants.dev_velo_module_zs
  );

  state.invoke();
}
