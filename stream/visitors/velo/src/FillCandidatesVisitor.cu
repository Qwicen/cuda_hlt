#include "SequenceVisitor.cuh"
#include "FillCandidates.cuh"

template<>
void SequenceVisitor::set_arguments_size<velo_fill_candidates_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_h0_candidates>(2 * host_buffers.host_total_number_of_velo_clusters[0]);
  arguments.set_size<dev_h2_candidates>(2 * host_buffers.host_total_number_of_velo_clusters[0]);
}

template<>
void SequenceVisitor::visit<velo_fill_candidates_t>(
  velo_fill_candidates_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts and arguments
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0], 48), dim3(128), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_velo_cluster_container>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_module_cluster_num>(),
    arguments.offset<dev_h0_candidates>(),
    arguments.offset<dev_h2_candidates>()
  );
  state.invoke();
}
