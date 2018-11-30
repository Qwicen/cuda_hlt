#include "SequenceVisitor.cuh"
#include "SciFiRawBankDecoderV4.cuh"

template<>
void SequenceVisitor::set_arguments_size<scifi_raw_bank_decoder_v4_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_scifi_hits>(host_buffers.scifi_hits_uints());
}

template<>
void SequenceVisitor::visit<scifi_raw_bank_decoder_v4_t>(
  scifi_raw_bank_decoder_v4_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_scifi_raw_input>(),
    arguments.offset<dev_scifi_raw_input_offsets>(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.offset<dev_scifi_hits>(),
    arguments.offset<dev_event_list>(),
    constants.dev_scifi_geometry,
    constants.dev_inv_clus_res
  );

  state.invoke();
}
