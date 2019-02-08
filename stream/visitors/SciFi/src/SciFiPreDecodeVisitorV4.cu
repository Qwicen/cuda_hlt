#include "SequenceVisitor.cuh"
#include "SciFiPreDecodeV4.cuh"

template<>
void SequenceVisitor::set_arguments_size<scifi_pre_decode_v4_t>(
  scifi_pre_decode_v4_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_scifi_hits>(host_buffers.scifi_hits_uints());
}

template<>
void SequenceVisitor::visit<scifi_pre_decode_v4_t>(
  scifi_pre_decode_v4_t& state,
  const scifi_pre_decode_v4_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(
    dim3(host_buffers.host_number_of_selected_events[0]), dim3(SciFi::SciFiRawBankParams::NbBanks), cuda_stream);
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
