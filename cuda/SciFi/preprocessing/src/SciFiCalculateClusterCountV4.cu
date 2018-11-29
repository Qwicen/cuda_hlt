#include "SciFiCalculateClusterCountV4.cuh"

using namespace SciFi;

__global__ void scifi_calculate_cluster_count_v4(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  uint* scifi_hit_count,
  char* scifi_geometry)
{
  const uint event_number = blockIdx.x;

  const SciFiRawEvent event(scifi_raw_input + scifi_raw_input_offsets[event_number]);
  const SciFiGeometry geom(scifi_geometry);
  SciFi::HitCount hit_count;
  hit_count.typecast_before_prefix_sum(scifi_hit_count, event_number);

  for (uint i = threadIdx.x; i < SciFi::Constants::n_consecutive_raw_banks; i += blockDim.x) {
    uint32_t* hits_mat = hit_count.n_hits_mats + i;
    const auto rawbank = event.getSciFiRawBank(i);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    if (*(last - 1) == 0) --last; // Remove padding at the end
    const uint number_of_clusters = last - it;

    atomicAdd(hits_mat, number_of_clusters);
  }

  const uint mats_difference = 3 * SciFi::Constants::n_consecutive_raw_banks;
  for (uint i = SciFi::Constants::n_consecutive_raw_banks + threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x) {
    uint32_t* hits_mat;
    const auto rawbank = event.getSciFiRawBank(i);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    // For details see RawBankDecoder
    if (*(last - 1) == 0) --last; // Remove phadding at the end
    for (; it < last; ++it) {     // loop over the clusters
      uint16_t c = *it;
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
      hits_mat = hit_count.n_hits_mats + SciFiChannelID(ch).correctedUniqueMat() - mats_difference;
      atomicAdd(hits_mat, 1);
    }
  }
}
