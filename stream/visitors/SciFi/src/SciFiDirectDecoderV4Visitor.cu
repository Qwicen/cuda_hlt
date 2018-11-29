#include "SequenceVisitor.cuh"
#include "SciFiDirectDecoderV4.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(scifi_direct_decoder_v4_t)

template<>
void SequenceVisitor::visit<scifi_direct_decoder_v4_t>(
  scifi_direct_decoder_v4_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(runtime_options.number_of_events), dim3(64, 64), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_scifi_raw_input>(),
    arguments.offset<dev_scifi_raw_input_offsets>(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.offset<dev_scifi_hits>(),
    constants.dev_scifi_geometry,
    constants.dev_inv_clus_res
  );

  state.invoke();
}
