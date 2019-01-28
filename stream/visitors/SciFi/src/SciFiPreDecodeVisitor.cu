#include "SequenceVisitor.cuh"
#include "SciFiPreDecode.cuh"

template<>
void SequenceVisitor::set_arguments_size<scifi_pre_decode_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_scifi_hits>(host_buffers.scifi_hits_uints());
}

template<>
void SequenceVisitor::visit<scifi_pre_decode_t>(
  scifi_pre_decode_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(SciFi::SciFiRawBankParams::NbBanks), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_scifi_raw_input>(),
    arguments.offset<dev_scifi_raw_input_offsets>(),
    arguments.offset<dev_event_list>(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.offset<dev_scifi_hits>(),
    constants.dev_scifi_geometry,
    constants.dev_inv_clus_res
  );

  state.invoke();
}
