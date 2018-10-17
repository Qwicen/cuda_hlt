#include "EstimateClusterCount.cuh"

using namespace SciFi;

__global__ void estimate_cluster_count(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  uint* scifi_hit_count, char* scifi_geometry
) {
  const uint event_number = blockIdx.x;

  const SciFiRawEvent event(scifi_raw_input + scifi_raw_input_offsets[event_number]);
  const SciFiGeometry geom(scifi_geometry);
  SciFiHitCount hit_count;
  hit_count.typecast_before_prefix_sum(scifi_hit_count, event_number);

  // NO version checking. Be careful, as v5 is assumed.
  // assert(event.version == 5u);

  for(uint rawbank = threadIdx.x; rawbank < geom.number_of_tell40s; rawbank += blockDim.x)
  {
    SciFiChannelID ch = geom.bank_first_channel[rawbank];
    // Apparently the unique* methods are not designed to start at 0, therefore -16
    uint32_t* hits_layer = hit_count.n_hits_layers + ((ch.uniqueQuarter() - 16) >> 1);
    // approx. 2 bytes per cluster in v5 (overestimates a little). See LHCb::FTDAQ::nbFTClusters (FTDAQHelper.cpp)
    uint32_t estimated_cluster_count = (event.raw_bank_offset[rawbank + 1] - event.raw_bank_offset[rawbank]) / 2 - 2u;
    atomicAdd(hits_layer, estimated_cluster_count);
  }
}
