#include "SequenceVisitor.cuh"
#include "UTPreDecode.cuh"

template<>
void SequenceVisitor::visit<ut_pre_decode_t>(
  ut_pre_decode_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_ut_hits>(UTHits::number_of_arrays * host_buffers.host_accumulated_number_of_ut_hits[0]);
  arguments.set_size<arg::dev_ut_hit_count>(runtime_options.number_of_events * constants.host_unique_x_sector_layer_offsets[4]);
  scheduler.setup_next(arguments, sequence_step);

  cudaCheck(cudaMemsetAsync(arguments.offset<arg::dev_ut_hit_count>(),
    0,
    arguments.size<arg::dev_ut_hit_count>(),
    cuda_stream));

  state.set_opts(dim3(runtime_options.number_of_events), dim3(64, 4), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_ut_raw_input>(),
    arguments.offset<arg::dev_ut_raw_input_offsets>(),
    constants.dev_ut_boards,
    constants.dev_ut_geometry,
    constants.dev_ut_region_offsets,
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_x_sector_offsets,
    arguments.offset<arg::dev_ut_hit_offsets>(),
    arguments.offset<arg::dev_ut_hits>(),
    arguments.offset<arg::dev_ut_hit_count>()
  );

  state.invoke();
}
