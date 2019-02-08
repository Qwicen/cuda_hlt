#include "SequenceVisitor.cuh"
#include "SciFiRawBankDecoderV4.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(scifi_raw_bank_decoder_v4_t)

template<>
void SequenceVisitor::visit<scifi_raw_bank_decoder_v4_t>(
  scifi_raw_bank_decoder_v4_t& state,
  const scifi_raw_bank_decoder_v4_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
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
    constants.dev_inv_clus_res);

  state.invoke();
}
