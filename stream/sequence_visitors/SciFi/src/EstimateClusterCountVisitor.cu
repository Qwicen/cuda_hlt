#include "StreamVisitor.cuh"

template<>
void StreamVisitor::visit<decltype(weak_tracks_adder_t(weak_tracks_adder))>(
  decltype(weak_tracks_adder_t(weak_tracks_adder))& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_scifi_raw_input>(runtime_options.host_scifi_events_size);
  arguments.set_size<arg::dev_scifi_raw_input_offsets>(runtime_options.host_scifi_event_offsets_size);
  arguments.set_size<arg::dev_scifi_hit_count>(2 * runtime_options.number_of_events * SciFi::number_of_zones + 1);
  scheduler.setup_next(arguments, sequence_step++);

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

  sequence.set_opts<seq::estimate_cluster_count>(dim3(runtime_options.number_of_events), dim3(240), cuda_stream);
  sequence.set_arguments<seq::estimate_cluster_count>(
    arguments.offset<arg::dev_scifi_raw_input>(),
    arguments.offset<arg::dev_scifi_raw_input_offsets>(),
    arguments.offset<arg::dev_scifi_hit_count>(),
    constants.dev_scifi_geometry
  );
  sequence.invoke<seq::estimate_cluster_count>();

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
}
