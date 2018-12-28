#include "SequenceVisitor.cuh"
#include "UTCalculateNumberOfHits.cuh"

template<>
void SequenceVisitor::set_arguments_size<ut_calculate_number_of_hits_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_ut_hit_offsets>(
    host_buffers.host_number_of_selected_events[0] * constants.host_unique_x_sector_layer_offsets[4] + 1);
}

template<>
void SequenceVisitor::visit<ut_calculate_number_of_hits_t>(
  ut_calculate_number_of_hits_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts and arguments for kernel call
  cudaCheck(
    cudaMemsetAsync(arguments.offset<dev_ut_hit_offsets>(), 0, arguments.size<dev_ut_hit_offsets>(), cuda_stream));

  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(64, 4), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_ut_raw_input>(),
    arguments.offset<dev_ut_raw_input_offsets>(),
    constants.dev_ut_boards,
    constants.dev_ut_region_offsets,
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_x_sector_offsets,
    arguments.offset<dev_ut_hit_offsets>(),
    arguments.offset<dev_event_list>());

  // Invoke kernel
  state.invoke();

  // // Print UT hit count per event per layer
  // std::vector<uint> host_ut_hit_offsets (arguments.size<dev_ut_hit_offsets>() / sizeof(uint));

  // info_cout << "variable: " << host_ut_hit_offsets.data() << ", " << host_ut_hit_offsets.size() << std::endl
  //   << "offset: " << arguments.offset<dev_ut_hit_offsets>() << ", "
  //   << "size: " << arguments.size<dev_ut_hit_offsets>() << std::endl;
  
  // cudaCheck(cudaMemcpy(
  //   host_ut_hit_offsets.data(),
  //   arguments.offset<dev_ut_hit_offsets>(),
  //   arguments.size<dev_ut_hit_offsets>(),
  //   cudaMemcpyDeviceToHost));

  // cudaEventRecord(cuda_generic_event, cuda_stream);
  // cudaEventSynchronize(cuda_generic_event);

  // for (int e = 0; e < host_buffers.host_number_of_selected_events[0]; ++e) {
  //   info_cout << "Event " << e << ", #hits per layer: ";
  //   uint32_t* count = host_ut_hit_offsets.data() + e * constants.host_unique_x_sector_layer_offsets[4];
  //   for (uint32_t i = 0; i < constants.host_unique_x_sector_layer_offsets[4]; ++i) { info_cout << count[i] << ", "; }
  //   info_cout << std::endl;
  // }
}
