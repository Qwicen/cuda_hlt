#include "SequenceVisitor.cuh"
#include "MaskedVeloClustering.cuh"

template<>
void SequenceVisitor::set_arguments_size<velo_masked_clustering_t>(
  velo_masked_clustering_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_velo_cluster_container>(6 * host_buffers.host_total_number_of_velo_clusters[0]);
}

template<>
void SequenceVisitor::visit<velo_masked_clustering_t>(
  velo_masked_clustering_t& state,
  const velo_masked_clustering_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_velo_raw_input>(),
    arguments.offset<dev_velo_raw_input_offsets>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_module_cluster_num>(),
    arguments.offset<dev_module_candidate_num>(),
    arguments.offset<dev_cluster_candidates>(),
    arguments.offset<dev_velo_cluster_container>(),
    arguments.offset<dev_event_list>(),
    arguments.offset<dev_event_order>(),
    constants.dev_velo_geometry,
    constants.dev_velo_sp_patterns,
    constants.dev_velo_sp_fx,
    constants.dev_velo_sp_fy);

  state.invoke();
}
