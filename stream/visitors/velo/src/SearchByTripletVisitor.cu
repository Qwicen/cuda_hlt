#include "SequenceVisitor.cuh"
#include "SearchByTriplet.cuh"

template<>
void SequenceVisitor::set_arguments_size<velo_search_by_triplet_t>(
  velo_search_by_triplet_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_tracks>(host_buffers.host_number_of_selected_events[0] * Velo::Constants::max_tracks);
  arguments.set_size<dev_tracklets>(host_buffers.host_number_of_selected_events[0] * Velo::Tracking::ttf_modulo);
  arguments.set_size<dev_tracks_to_follow>(host_buffers.host_number_of_selected_events[0] * Velo::Tracking::ttf_modulo);
  arguments.set_size<dev_weak_tracks>(host_buffers.host_number_of_selected_events[0] * Velo::Tracking::max_weak_tracks);
  arguments.set_size<dev_hit_used>(host_buffers.host_total_number_of_velo_clusters[0]);
  arguments.set_size<dev_atomics_velo>(host_buffers.host_number_of_selected_events[0] * Velo::num_atomics);
  arguments.set_size<dev_rel_indices>(host_buffers.host_number_of_selected_events[0] * 2 * Velo::Constants::max_numhits_in_module);
}

template<>
void SequenceVisitor::visit<velo_search_by_triplet_t>(
  velo_search_by_triplet_t& state,
  const velo_search_by_triplet_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts and arguments
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(32), cuda_stream, 32 * sizeof(float));
  state.set_arguments(
    arguments.offset<dev_velo_cluster_container>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_module_cluster_num>(),
    arguments.offset<dev_tracks>(),
    arguments.offset<dev_tracklets>(),
    arguments.offset<dev_tracks_to_follow>(),
    arguments.offset<dev_weak_tracks>(),
    arguments.offset<dev_hit_used>(),
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_h0_candidates>(),
    arguments.offset<dev_h2_candidates>(),
    arguments.offset<dev_rel_indices>(),
    constants.dev_velo_module_zs
  );

  state.invoke();
}
