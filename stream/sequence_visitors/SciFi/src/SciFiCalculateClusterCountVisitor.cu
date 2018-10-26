#include "SequenceVisitor.cuh"
#include "SciFiCalculateClusterCount.cuh"

template<>
void SequenceVisitor::visit<scifi_calculate_cluster_count_t>(
  scifi_calculate_cluster_count_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_scifi_raw_input>(runtime_options.host_scifi_events_size);
  arguments.set_size<arg::dev_scifi_raw_input_offsets>(runtime_options.host_scifi_event_offsets_size);
  arguments.set_size<arg::dev_scifi_hit_count>(2 * runtime_options.number_of_events * SciFi::number_of_mats + 1);
  scheduler.setup_next(arguments, sequence_step);

  cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_scifi_raw_input>(),
    runtime_options.host_scifi_events,
    runtime_options.host_scifi_events_size,
    cudaMemcpyHostToDevice,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_scifi_raw_input_offsets>(),
    runtime_options.host_scifi_event_offsets,
    runtime_options.host_scifi_event_offsets_size * sizeof(uint),
    cudaMemcpyHostToDevice,
    cuda_stream));

  cudaCheck(cudaMemsetAsync(arguments.offset<arg::dev_scifi_hit_count>(),
    0,
    arguments.size<arg::dev_scifi_hit_count>(),
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  state.set_opts(dim3(runtime_options.number_of_events), dim3(240), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_scifi_raw_input>(),
    arguments.offset<arg::dev_scifi_raw_input_offsets>(),
    arguments.offset<arg::dev_scifi_hit_count>(),
    constants.dev_scifi_geometry
  );

  state.invoke();
}
