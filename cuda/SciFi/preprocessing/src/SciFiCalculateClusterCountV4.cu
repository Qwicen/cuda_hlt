#include "SciFiCalculateClusterCountV4.cuh"

using namespace SciFi;

/**
 * @brief This function calculates the amount of clusters in all mats.
 * @details More details about the SciFi format:
 *          https://cds.cern.ch/record/2630154/files/LHCb-INT-2018-024.pdf
 */
__global__ void scifi_calculate_cluster_count_v4(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  uint* scifi_hit_count,
  const uint* event_list,
  char* scifi_geometry)
{
  const uint event_number = blockIdx.x;
  const uint selected_event_number = event_list[event_number];
  
  const SciFiRawEvent event(scifi_raw_input + scifi_raw_input_offsets[selected_event_number]);
  const SciFiGeometry geom(scifi_geometry);
  SciFi::HitCount hit_count {scifi_hit_count, event_number};

  for (uint i = threadIdx.x; i < SciFi::Constants::n_consecutive_raw_banks; i += blockDim.x) {
    const uint k = i % 10;
    const bool reverse_raw_bank_order = k < 5;
    const uint current_raw_bank = reverse_raw_bank_order ?
      5 * (i / 5) + (4 - i % 5) :
      i;

    const auto rawbank = event.getSciFiRawBank(current_raw_bank);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    if (*(last - 1) == 0) --last; // Remove padding at the end
    const uint number_of_clusters = last - it;

    if (last > it) {
      hit_count.mat_offsets[i] = number_of_clusters;
    }
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
      hits_mat = hit_count.mat_offsets + SciFiChannelID(ch).correctedUniqueMat() - mats_difference;
      atomicAdd(hits_mat, 1);
    }
  }
}
