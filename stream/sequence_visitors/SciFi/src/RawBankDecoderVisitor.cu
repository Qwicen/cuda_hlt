#include "SequenceVisitor.cuh"
#include "RawBankDecoder.cuh"

template<>
void SequenceVisitor::visit<raw_bank_decoder_t>(
  raw_bank_decoder_t& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  arguments.set_size<arg::dev_scifi_hits>(host_buffers.scifi_hits_bytes());
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events), dim3(240), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_scifi_raw_input>(),
    arguments.offset<arg::dev_scifi_raw_input_offsets>(),
    arguments.offset<arg::dev_scifi_hit_count>(),
    arguments.offset<arg::dev_scifi_hits>(),
    constants.dev_scifi_geometry
  );

  state.invoke();
  
  // // SciFi Decoder Debugging
  // const uint hit_count_uints = 2 * number_of_events * SciFi::number_of_zones + 1;
  // uint host_scifi_hit_count[hit_count_uints];
  // char* host_scifi_hits = new char[hits_bytes];
  // uint* host_scifi_hit_permutation = new uint[*host_accumulated_number_of_scifi_hits];
  // cudaCheck(cudaMemcpyAsync(&host_scifi_hit_count, arguments.offset<arg::dev_scifi_hit_count>(), hit_count_uints*sizeof(uint), cudaMemcpyDeviceToHost, stream));
  // cudaCheck(cudaMemcpyAsync(host_scifi_hits, arguments.offset<arg::dev_scifi_hits>(), arguments.size<hits_bytes>(), cudaMemcpyDeviceToHost, stream));
  // cudaCheck(cudaMemcpyAsync(host_scifi_hit_permutation, arguments.offset<arg::dev_scifi_hit_permutations>(), arguments.offset<arg::dev_scifi_hit_permutations>(), cudaMemcpyDeviceToHost, stream));
  // cudaEventRecord(cuda_generic_event, stream);
  // cudaEventSynchronize(cuda_generic_event);

  // SciFi::SciFiHits host_scifi_hits_struct;
  // host_scifi_hits_struct.typecast_sorted(host_scifi_hits, host_scifi_hit_count[number_of_events * SciFi::number_of_zones]);

  // //Print only non-empty hits
  // std::ofstream outfile("dump.txt");
  // SciFi::SciFiHitCount host_scifi_hit_count_struct;
  // for(size_t event = 0; event < number_of_events; event++) {
  //   host_scifi_hit_count_struct.typecast_ascifier_prefix_sum(host_scifi_hit_count, event, number_of_events);
  //   for(size_t zone = 0; zone < SciFi::number_of_zones; zone++) {
  //     for(size_t hit = 0; hit < host_scifi_hit_count_struct.n_hits_layers[zone]; hit++) {
  //       auto h = host_scifi_hits_struct.getHit(host_scifi_hit_count_struct.layer_offsets[zone] + hit);
  //       outfile << std::setprecision(8) << std::fixed << h.planeCode << " " << h.hitZone << " " << h.LHCbID << " "
  //         << h.x0 << " " << h.z0 << " " << h.w<< " " << h.dxdy << " "
  //         << h.dzdy << " " << h.yMin << " " << h.yMax  <<  std::endl;
  //     }
  //   }
  // }
}
