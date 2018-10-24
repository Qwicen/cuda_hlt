#include "SciFiPreDecode.cuh"
#include "assert.h"

using namespace SciFi;

__device__ void store_sorted_cluster_reference (
  const SciFiHitCount& hit_count,
  const uint32_t uniqueMat,
  const uint32_t chan,
  const uint32_t* shared_mat_offsets,
  const int raw_bank,
  const int it,
  const int condition_1,
  const int condition_2,
  const int delta,
  SciFiHits& hits)
{
  uint32_t* hits_mat = hit_count.n_hits_mats + uniqueMat;
  uint32_t hitIndex = (*hits_mat)++;

  const SciFi::SciFiChannelID id {chan};
  if (id.reversedZone()) {
    hitIndex = hit_count.mat_number_of_hits(uniqueMat) - 1 - hitIndex;
  }

  assert(hitIndex < hit_count.mat_number_of_hits(uniqueMat));
  hitIndex += shared_mat_offsets[uniqueMat];

  // Cluster reference:
  //   raw bank: 8 bits
  //   element (it): 8 bits
  //   Condition 1-2-3: 2 bits
  //   Condition 2.1-2.2: 1 bit
  //   Condition 2.1: log2(n+1) - 8 bits
  hits.cluster_reference[hitIndex] = (raw_bank & 0xFF) << 24
    | (it & 0xFF) << 16
    | (condition_1 & 0x03) << 14
    | (condition_2 & 0x01) << 13
    | (delta & 0xFF);
};

__global__ void scifi_pre_decode(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  char *scifi_hits,
  char *scifi_geometry
) {
  const int number_of_events = gridDim.x;
  const int event_number = blockIdx.x;

  // maybe not hardcoded, or in another place
  const float invClusRes[] = {1/0.05, 1/0.08, 1/0.11, 1/0.14, 1/0.17, 1/0.20, 1/0.23, 1/0.26, 1/0.29};

  SciFiGeometry geom(scifi_geometry);
  const auto event = SciFiRawEvent(scifi_events + scifi_event_offsets[event_number]);

  SciFiHits hits;
  hits.typecast_unsorted(scifi_hits, scifi_hit_count[number_of_events * SciFi::number_of_mats]);
  SciFiHitCount hit_count;
  hit_count.typecast_after_prefix_sum(scifi_hit_count, event_number, number_of_events);

  __shared__ uint32_t shared_mat_offsets[SciFi::number_of_mats];

  for (uint i = threadIdx.x; i < SciFi::number_of_mats; i += blockDim.x) {
    shared_mat_offsets[i] = hit_count.mat_offsets[i];
    hit_count.n_hits_mats[i] = 0;
  }

  __syncthreads();

  // Main execution loop
  for(uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x) {
    auto rawbank = event.getSciFiRawBank(i);
    const uint16_t* starting_it = rawbank.data + 2;
    uint16_t* last = rawbank.last;
    if (*(last-1) == 0) --last; // Remove padding at the end

    if (starting_it < last) {
      const uint number_of_iterations = last - starting_it;
      for (uint it_number=0; it_number<number_of_iterations; ++it_number){
        auto it = starting_it + it_number;
        const uint16_t c = *it;
        const uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
        const auto chid = SciFiChannelID(ch);
        const uint32_t correctedMat = chid.correctedUniqueMat();

        // Condition 1: "00"
        if (!cSize(c) || it+1 == last) {
          const int condition_1 = 0x00;

          store_sorted_cluster_reference (
            hit_count,
            correctedMat,
            ch,
            (const uint32_t*) &shared_mat_offsets[0],
            i,
            it_number,
            condition_1, // Condition 1
            0x00, // Condition 2
            0x00, // Delta
            hits);
        } else {
          const unsigned c2 = *(it+1);
          if (cSize(c2) && getLinkInBank(c) == getLinkInBank(c2)) {
            // Condition 1: "01"
            const int condition_1 = 0x01;

            const auto delta = (cell(c2) - cell(c));
            if (delta  > SciFiRawBankParams::clusterMaxWidth) {
              // Condition 2: "0"
              const int condition_2 = 0x00;

              for (auto j = SciFiRawBankParams::clusterMaxWidth; j < delta; j += SciFiRawBankParams::clusterMaxWidth){
                // Delta equals j / SciFiRawBankParams::clusterMaxWidth
                const int delta_parameter = j / SciFiRawBankParams::clusterMaxWidth;
                store_sorted_cluster_reference (
                  hit_count,
                  correctedMat,
                  ch,
                  (const uint32_t*) &shared_mat_offsets[0],
                  i,
                  it_number,
                  condition_1,
                  condition_2,
                  delta_parameter,
                  hits);
              }

              // Delta equals 0
              const int delta_parameter = 0;
              store_sorted_cluster_reference (
                hit_count,
                correctedMat,
                ch,
                (const uint32_t*) &shared_mat_offsets[0],
                i,
                it_number,
                condition_1,
                condition_2,
                delta_parameter,
                hits);
            } else {
              // Condition 2: "1"
              const int condition_2 = 0x01;

              const auto widthClus = 2 * delta - 1 + fraction(c2);
              store_sorted_cluster_reference (
                hit_count,
                correctedMat,
                ch,
                (const uint32_t*) &shared_mat_offsets[0],
                i,
                it_number,
                condition_1,
                condition_2,
                0x00,
                hits);
            }

            // Due to v5
            ++it_number;
          } else {
            // Condition 1: "10"
            const int condition_1 = 0x02;
            
            store_sorted_cluster_reference (
              hit_count,
              correctedMat,
              ch,
              (const uint32_t*) &shared_mat_offsets[0],
              i,
              it_number,
              condition_1,
              0x00,
              0x00,
              hits);
          }
        }
      }
    }
  }
}
