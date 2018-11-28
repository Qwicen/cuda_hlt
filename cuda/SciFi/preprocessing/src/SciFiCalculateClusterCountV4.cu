#include "SciFiCalculateClusterCountV4.cuh"

using namespace SciFi;

__global__ void scifi_calculate_cluster_count_v4(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  uint* scifi_hit_count,
  char* scifi_geometry
) {
  const uint event_number = blockIdx.x;

  const SciFiRawEvent event(scifi_raw_input + scifi_raw_input_offsets[event_number]);
  const SciFiGeometry geom(scifi_geometry);
  SciFi::HitCount hit_count;
  hit_count.typecast_before_prefix_sum(scifi_hit_count, event_number);

  // NO version checking. Be careful, as v5 is assumed.

  for(uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x)
  {
    uint32_t* hits_mat;
    const auto rawbank = event.getSciFiRawBank(i);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    //For details see RawBankDecoder
    if (*(last-1) == 0) --last; //Remove phadding at the end
    for( ;  it < last; ++it ) { // loop over the clusters
      uint16_t c = *it;
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
      hits_mat = hit_count.n_hits_mats + SciFiChannelID(ch).correctedUniqueMat();
      atomicAdd(hits_mat, 1);
    }
  }
}
