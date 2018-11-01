#include "SequenceVisitor.cuh"
#include "UTCalculateNumberOfHits.cuh"

template<>
void SequenceVisitor::visit<ut_calculate_number_of_hits_t>(
  ut_calculate_number_of_hits_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Set arguments and reserve memory
  arguments.set_size<arg::dev_ut_raw_input>(runtime_options.host_ut_events_size);
  arguments.set_size<arg::dev_ut_raw_input_offsets>(runtime_options.host_ut_event_offsets_size);
  arguments.set_size<arg::dev_ut_hit_offsets>(runtime_options.number_of_events * constants.host_unique_x_sector_layer_offsets[4] + 1);
  scheduler.setup_next(arguments, sequence_step);

  // Setup opts and arguments for kernel call
  cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_ut_raw_input>(),
    runtime_options.host_ut_events,
    runtime_options.host_ut_events_size,
    cudaMemcpyHostToDevice,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_ut_raw_input_offsets>(),
    runtime_options.host_ut_event_offsets,
    runtime_options.host_ut_event_offsets_size * sizeof(uint32_t),
    cudaMemcpyHostToDevice,
    cuda_stream));

  cudaCheck(cudaMemsetAsync(arguments.offset<arg::dev_ut_hit_offsets>(),
    0,
    arguments.size<arg::dev_ut_hit_offsets>(),
    cuda_stream));

  state.set_opts(dim3(runtime_options.number_of_events), dim3(64, 4), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_ut_raw_input>(),
    arguments.offset<arg::dev_ut_raw_input_offsets>(),
    constants.dev_ut_boards,
    constants.dev_ut_region_offsets,
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_x_sector_offsets,
    arguments.offset<arg::dev_ut_hit_offsets>()
  );

  // Invoke kernel
  state.invoke();

  // // Print UT hit count per event per layer
  // std::vector<uint> host_ut_hit_count (number_of_events * constants.host_unique_x_sector_layer_offsets[4] + 1);
  // cudaCheck(cudaMemcpyAsync(host_ut_hit_count.data(), arguments.offset<arg::dev_ut_hit_offsets>(), argen.size<arg::dev_ut_hit_offsets>(number_of_events * constants.host_unique_x_sector_layer_offsets[4] + 1), cudaMemcpyDeviceToHost, stream));
  // cudaEventRecord(cuda_generic_event, stream);
  // cudaEventSynchronize(cuda_generic_event);
  // for (int e=0; e<number_of_events; ++e) {
  //   info_cout << "Event " << e << ", #hits per layer: ";
  //   uint32_t* count = host_ut_hit_count.data() + e * constants.host_unique_x_sector_layer_offsets[4];
  //   for (uint32_t i = 0; i < constants.host_unique_x_sector_layer_offsets[4]; ++i) {
  //     info_cout << count[i] << ", ";
  //   }
  //   info_cout << std::endl;
  // }
}
