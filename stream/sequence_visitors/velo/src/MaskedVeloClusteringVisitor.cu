#include "SequenceVisitor.cuh"
#include "MaskedVeloClustering.cuh"

template<>
void SequenceVisitor::visit<masked_velo_clustering_t>(
  masked_velo_clustering_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_velo_cluster_container>(6 * host_buffers.host_total_number_of_velo_clusters[0]);
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_raw_input>(),
    arguments.offset<arg::dev_raw_input_offsets>(),
    arguments.offset<arg::dev_estimated_input_size>(),
    arguments.offset<arg::dev_module_cluster_num>(),
    arguments.offset<arg::dev_module_candidate_num>(),
    arguments.offset<arg::dev_cluster_candidates>(),
    arguments.offset<arg::dev_velo_cluster_container>(),
    constants.dev_velo_geometry,
    constants.dev_velo_sp_patterns,
    constants.dev_velo_sp_fx,
    constants.dev_velo_sp_fy
  );
  
  state.invoke();
}
