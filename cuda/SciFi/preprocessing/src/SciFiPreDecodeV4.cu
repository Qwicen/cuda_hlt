#include "SciFiPreDecodeV4.cuh"
#include "assert.h"

using namespace SciFi;

__device__ void store_sorted_cluster_reference_v4(
  const SciFi::HitCount& hit_count,
  const uint32_t uniqueMat,
  const uint32_t chan,
  const uint32_t* shared_mat_offsets,
  uint32_t* shared_mat_count,
  const int raw_bank,
  const int it,
  SciFi::Hits& hits)
{
  uint32_t hitIndex = (*shared_mat_count)++;

  const SciFi::SciFiChannelID id {chan};
  if (id.reversedZone()) {
    hitIndex = hit_count.mat_number_of_hits(uniqueMat) - 1 - hitIndex;
  }

  assert(hitIndex < hit_count.mat_number_of_hits(uniqueMat));
  hitIndex += *shared_mat_offsets;

  // Cluster reference:
  //   raw bank: 8 bits
  //   element (it): 8 bits
  hits.cluster_reference[hitIndex] = (raw_bank & 0xFF) << 8 | (it & 0xFF);
};

__global__ void scifi_pre_decode_v4(
  char* scifi_events,
  uint* scifi_event_offsets,
  uint* scifi_hit_count,
  uint* scifi_hits,
  const uint* event_list,
  char* scifi_geometry,
  const float* dev_inv_clus_res)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint selected_event_number = event_list[event_number];

  SciFiGeometry geom(scifi_geometry);
  const auto event = SciFiRawEvent(scifi_events + scifi_event_offsets[selected_event_number]);

  SciFi::Hits hits {
    scifi_hits, scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats], &geom, dev_inv_clus_res};
  const SciFi::HitCount hit_count {scifi_hit_count, event_number};

  __shared__ uint32_t shared_mat_offsets[SciFi::Constants::n_mats_without_group];
  __shared__ uint32_t shared_mat_count[SciFi::Constants::n_mats_without_group];
  for (uint i = threadIdx.x; i < SciFi::Constants::n_mats_without_group; i += blockDim.x) {
    shared_mat_offsets[i] = hit_count.mat_offset(
      i + SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank);
    shared_mat_count[i] = 0;
  }

  __syncthreads();

  // Main execution loop
  for (uint i = SciFi::Constants::n_consecutive_raw_banks + threadIdx.x; i < event.number_of_raw_banks;
       i += blockDim.x) {
    auto rawbank = event.getSciFiRawBank(i);
    const uint16_t* starting_it = rawbank.data + 2;
    uint16_t* last = rawbank.last;
    if (*(last - 1) == 0) --last; // Remove padding at the end

    if (starting_it < last) {
      const uint number_of_iterations = last - starting_it;
      for (uint it_number = 0; it_number < number_of_iterations; ++it_number) {
        auto it = starting_it + it_number;
        const uint16_t c = *it;
        const uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
        const auto chid = SciFiChannelID(ch);
        const uint32_t correctedMat = chid.correctedUniqueMat();

        store_sorted_cluster_reference_v4(
          hit_count,
          correctedMat,
          ch,
          (const uint32_t*) &shared_mat_offsets
            [correctedMat - SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank],
          (uint32_t*) &shared_mat_count
            [correctedMat - SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank],
          i,
          it_number,
          hits);
      }
    }
  }
}
