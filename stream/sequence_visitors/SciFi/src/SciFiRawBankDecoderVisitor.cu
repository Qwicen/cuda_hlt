#include "SequenceVisitor.cuh"
#include "SciFiRawBankDecoder.cuh"

template<>
void SequenceVisitor::visit<scifi_raw_bank_decoder_t>(
  scifi_raw_bank_decoder_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_scifi_hits>(host_buffers.scifi_hits_uints());
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_scifi_raw_input>(),
    arguments.offset<arg::dev_scifi_raw_input_offsets>(),
    arguments.offset<arg::dev_scifi_hit_count>(),
    arguments.offset<arg::dev_scifi_hits>(),
    constants.dev_scifi_geometry
  );

  state.invoke();

  // SciFi Decoder Debugging
  const uint hit_count_uints = 2 * runtime_options.number_of_events * SciFi::number_of_mats + 1;
  uint host_scifi_hit_count[hit_count_uints];
  uint* host_scifi_hits = new uint[host_buffers.scifi_hits_uints()];
  cudaCheck(cudaMemcpyAsync(&host_scifi_hit_count, arguments.offset<arg::dev_scifi_hit_count>(), hit_count_uints*sizeof(uint), cudaMemcpyDeviceToHost, cuda_stream));
  cudaCheck(cudaMemcpyAsync(host_scifi_hits, arguments.offset<arg::dev_scifi_hits>(), arguments.size<arg::dev_scifi_hits>(), cudaMemcpyDeviceToHost, cuda_stream));
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  SciFi::SciFiGeometry host_geom(constants.host_scifi_geometry);
  SciFi::SciFiHits hi(host_scifi_hits, host_scifi_hit_count[runtime_options.number_of_events * SciFi::number_of_mats], &host_geom);

  std::ofstream outfile("dump.txt");
  SciFi::SciFiHitCount host_scifi_hit_count_struct;
  for(size_t event = 0; event < runtime_options.number_of_events; event++) {
    host_scifi_hit_count_struct.typecast_after_prefix_sum(host_scifi_hit_count, event, runtime_options.number_of_events);
    for(size_t zone = 0; zone < SciFi::number_of_zones; zone++) {
      for(size_t hit = 0; hit < host_scifi_hit_count_struct.zone_number_of_hits(zone); hit++) {
        uint h = host_scifi_hit_count_struct.zone_offset(zone) + hit;
        outfile << std::setprecision(8) << std::fixed
          << hi.planeCode(h) << " "
          << zone % 2     << " "
          << hi.LHCbID(h) << " "
          << hi.x0[h]   << " "
          << hi.z0[h]   << " "
          << hi.w(h)    << " "
          << hi.dxdy(h) << " "
          << hi.dzdy(h) << " "
          << hi.yMin(h) << " "
          << hi.yMax(h) << std::endl;
      }
    }
  }
}
