#include "SequenceVisitor.cuh"
#include "UTFindPermutation.cuh"

template<>
void SequenceVisitor::visit<ut_find_permutation_t>(
  ut_find_permutation_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_ut_hit_permutations>(host_buffers.host_accumulated_number_of_ut_hits[0]);
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events, constants.host_unique_x_sector_layer_offsets[4]), dim3(16), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_ut_hits>(),
    arguments.offset<arg::dev_ut_hit_offsets>(),
    arguments.offset<arg::dev_ut_hit_permutations>(),
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_x_sector_offsets,
    constants.dev_unique_sector_xs
  );

  state.invoke();
}
