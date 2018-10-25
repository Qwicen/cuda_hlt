#include "SequenceVisitor.cuh"
#include "PrForward.cuh"

template<>
void SequenceVisitor::visit<scifi_pr_forward_t>(
  scifi_pr_forward_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_scifi_tracks>(runtime_options.number_of_events * SciFi::max_tracks);
  arguments.set_size<arg::dev_n_scifi_tracks>(runtime_options.number_of_events);
  scheduler.setup_next(arguments, sequence_step);
  
  state.set_opts(dim3(runtime_options.number_of_events), dim3(32), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_scifi_hits>(),
    arguments.offset<arg::dev_scifi_hit_count>(),
    arguments.offset<arg::dev_atomics_storage>(),
    arguments.offset<arg::dev_velo_track_hit_number>(),
    arguments.offset<arg::dev_velo_states>(),
    arguments.offset<arg::dev_veloUT_tracks>(),
    arguments.offset<arg::dev_atomics_veloUT>(),
    arguments.offset<arg::dev_scifi_tracks>(),
    arguments.offset<arg::dev_n_scifi_tracks>(),
    constants.dev_scifi_tmva1,
    constants.dev_scifi_tmva2,
    constants.dev_scifi_constArrays
  );
  state.invoke(); 
  
  // Transmission device to host
  // SciFi tracks
  cudaCheck(cudaMemcpyAsync(host_buffers.host_n_scifi_tracks,
    arguments.offset<arg::dev_n_scifi_tracks>(),
    arguments.size<arg::dev_n_scifi_tracks>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(host_buffers.host_scifi_tracks,
    arguments.offset<arg::dev_scifi_tracks>(),
    arguments.size<arg::dev_scifi_tracks>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));
}
