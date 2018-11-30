#include "SequenceVisitor.cuh"
#include "UTPreDecode.cuh"

template<>
void SequenceVisitor::set_arguments_size<ut_pre_decode_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_ut_hits>(UT::Hits::number_of_arrays * host_buffers.host_accumulated_number_of_ut_hits[0]);
  arguments.set_size<dev_ut_hit_count>(host_buffers.host_number_of_selected_events[0] * constants.host_unique_x_sector_layer_offsets[4]);
}

template<>
void SequenceVisitor::visit<ut_pre_decode_t>(
  ut_pre_decode_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  cudaCheck(cudaMemsetAsync(arguments.offset<dev_ut_hit_count>(),
    0,
    arguments.size<dev_ut_hit_count>(),
    cuda_stream));

  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(64, 4), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_ut_raw_input>(),
    arguments.offset<dev_ut_raw_input_offsets>(),
    arguments.offset<dev_event_list>(),
    constants.dev_ut_boards,
    constants.dev_ut_geometry,
    constants.dev_ut_region_offsets,
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_x_sector_offsets,
    arguments.offset<dev_ut_hit_offsets>(),
    arguments.offset<dev_ut_hits>(),
    arguments.offset<dev_ut_hit_count>()
  );

  state.invoke();
}
