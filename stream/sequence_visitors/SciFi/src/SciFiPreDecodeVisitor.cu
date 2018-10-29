#include "SequenceVisitor.cuh"
#include "SciFiPreDecode.cuh"

template<>
void SequenceVisitor::visit<scifi_pre_decode_t>(
  scifi_pre_decode_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_scifi_hits>(host_buffers.scifi_hits_uints());
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events), dim3(240), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_scifi_raw_input>(),
    arguments.offset<arg::dev_scifi_raw_input_offsets>(),
    arguments.offset<arg::dev_scifi_hit_count>(),
    arguments.offset<arg::dev_scifi_hits>(),
    constants.dev_scifi_geometry
  );

  state.invoke();
}
