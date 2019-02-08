#include "SequenceVisitor.cuh"
#include "InitEventList.cuh"

template<>
void SequenceVisitor::set_arguments_size<init_event_list_t>(
  init_event_list_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_velo_raw_input>(runtime_options.host_velopix_events_size);
  arguments.set_size<dev_velo_raw_input_offsets>(runtime_options.host_velopix_event_offsets_size);
  arguments.set_size<dev_ut_raw_input>(runtime_options.host_ut_events_size);
  arguments.set_size<dev_ut_raw_input_offsets>(runtime_options.host_ut_event_offsets_size);
  arguments.set_size<dev_scifi_raw_input>(runtime_options.host_scifi_events_size);
  arguments.set_size<dev_scifi_raw_input_offsets>(runtime_options.host_scifi_event_offsets_size);
  arguments.set_size<dev_event_list>(runtime_options.number_of_events);
  arguments.set_size<dev_number_of_selected_events>(1);
}

template<>
void SequenceVisitor::visit<init_event_list_t>(
  init_event_list_t& state,
  const init_event_list_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  assert(runtime_options.number_of_events > 0);

  // Setup opts and arguments for kernel call
  state.set_opts(
    dim3((runtime_options.number_of_events - 1) / 1024 + 1),
    dim3(runtime_options.number_of_events % 1024),
    cuda_stream);

  state.set_arguments(arguments.offset<dev_event_list>());

  // Fetch required arguments for the global event cuts algorithm and
  // the various decoding algorithms
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_velo_raw_input>(),
    runtime_options.host_velopix_events,
    arguments.size<dev_velo_raw_input>(),
    cudaMemcpyHostToDevice,
    cuda_stream));
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_velo_raw_input_offsets>(),
    runtime_options.host_velopix_event_offsets,
    arguments.size<dev_velo_raw_input_offsets>(),
    cudaMemcpyHostToDevice,
    cuda_stream));
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_ut_raw_input>(),
    runtime_options.host_ut_events,
    runtime_options.host_ut_events_size,
    cudaMemcpyHostToDevice,
    cuda_stream));
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_ut_raw_input_offsets>(),
    runtime_options.host_ut_event_offsets,
    runtime_options.host_ut_event_offsets_size * sizeof(uint32_t),
    cudaMemcpyHostToDevice,
    cuda_stream));
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_scifi_raw_input>(),
    runtime_options.host_scifi_events,
    runtime_options.host_scifi_events_size,
    cudaMemcpyHostToDevice,
    cuda_stream));
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_scifi_raw_input_offsets>(),
    runtime_options.host_scifi_event_offsets,
    runtime_options.host_scifi_event_offsets_size * sizeof(uint),
    cudaMemcpyHostToDevice,
    cuda_stream));

  state.invoke();

  // TODO: This is not needed here
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_event_list,
    arguments.offset<dev_event_list>(),
    runtime_options.number_of_events * sizeof(uint),
    cudaMemcpyDeviceToHost,
    cuda_stream));
}
