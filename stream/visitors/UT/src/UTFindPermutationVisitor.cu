#include "SequenceVisitor.cuh"
#include "UTFindPermutation.cuh"

template<>
void SequenceVisitor::set_arguments_size<ut_find_permutation_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_ut_hit_permutations>(host_buffers.host_accumulated_number_of_ut_hits[0]);
}

template<>
void SequenceVisitor::visit<ut_find_permutation_t>(
  ut_find_permutation_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0], constants.host_unique_x_sector_layer_offsets[4]), dim3(16), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_ut_hits>(),
    arguments.offset<dev_ut_hit_offsets>(),
    arguments.offset<dev_ut_hit_permutations>(),
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_x_sector_offsets,
    constants.dev_unique_sector_xs
  );

  state.invoke();
}
