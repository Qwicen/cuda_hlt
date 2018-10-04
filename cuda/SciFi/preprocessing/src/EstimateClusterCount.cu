#include "EstimateClusterCount.cuh"
#include "RawBankDecoder.cuh"

using namespace SciFi;

__global__ void estimate_cluster_count(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  uint* scifi_hit_count,
  char* scifi_geometry
) {
  const uint event_number = blockIdx.x;

  const SciFiRawEvent event(scifi_raw_input + scifi_raw_input_offsets[event_number]);
  const SciFiGeometry geom(scifi_geometry);
  SciFiHitCount hit_count;
  hit_count.typecast_before_prefix_sum(scifi_hit_count, event_number);

  // NO version checking. Be careful, as v5 is assumed.

  for(uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x)
  {
    uint32_t cluster_count = 0;
    const auto rawbank = event.getSciFiRawBank(i);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    //For details see RawBankDecoder
    if (*(last-1) == 0) --last; //Remove phadding at the end
    for( ;  it < last; ++it ){ // loop over the clusters
      uint16_t c = *it;
      if( !cSize(c) || it+1 == last ) { //No size flag or last cluster
        cluster_count++;
      } else { //Flagged or not the last one.
        unsigned c2 = *(it+1);
        if( cSize(c2) && getLinkInBank(c) == getLinkInBank(c2) ) {
          unsigned int delta = (cell(c2) - cell(c));
          cluster_count += 1 + (delta - 1) / SciFiRawBankParams::clusterMaxWidth;
          ++it;
        } else {
          cluster_count++;
        }
      }
    }

    const SciFi::SciFiChannelID id(geom.bank_first_channel[rawbank.sourceID]);
    uint32_t* const hits_layer = hit_count.n_hits_layers + ((id.uniqueQuarter() - 16) >> 1);
    atomicAdd(hits_layer, cluster_count);
  }
}
