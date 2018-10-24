#include "SciFiRawBankDecoder.cuh"
#include "assert.h"

using namespace SciFi;

__global__ void scifi_raw_bank_decoder(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  char *scifi_hits,
  char *scifi_geometry
) {
  // maybe not hardcoded, or in another place
  const float invClusRes[] = {1/0.05, 1/0.08, 1/0.11, 1/0.14, 1/0.17, 1/0.20, 1/0.23, 1/0.26, 1/0.29};
  const uint32_t number_of_events = gridDim.x;
  const uint32_t event_number = blockIdx.x;

  SciFiGeometry geom(scifi_geometry);
  const auto event = SciFiRawEvent(scifi_events + scifi_event_offsets[event_number]);

  SciFiHits hits;
  hits.typecast_unsorted(scifi_hits, scifi_hit_count[number_of_events * SciFi::number_of_mats]);
  SciFiHitCount hit_count;
  hit_count.typecast_after_prefix_sum(scifi_hit_count, event_number, number_of_events);

  __shared__ uint32_t shared_mat_offsets[SciFi::number_of_mats];

  for (uint i = threadIdx.x; i < SciFi::number_of_mats; i += blockDim.x) {
    shared_mat_offsets[i] = hit_count.mat_offsets[i];
  }

  for (uint i = threadIdx.x; i < SciFi::number_of_mats; i += blockDim.x) {
    hit_count.n_hits_mats[i] = 0;
  }

  __syncthreads();

  // Merge of PrStoreFTHit and RawBankDecoder.
  auto make_cluster = [&](uint32_t chan, uint8_t fraction, uint8_t pseudoSize, uint32_t uniqueMat) {
    const SciFi::SciFiChannelID id(chan);

    // Offset to save space in geometry structure, see DumpFTGeometry.cpp
    const uint32_t mat = id.uniqueMat() - 512;
    const uint32_t iQuarter = id.uniqueQuarter() - 16;
    const uint32_t planeCode = id.uniqueLayer() - 4;
    // See Kernel/LHCbID.h. Maybe no hardcoding?
    const uint32_t lhcbid = (10u << 28) + chan;
    const float dxdy = geom.dxdy[mat];
    const float dzdy = geom.dzdy[mat];
    const float globaldy = geom.globaldy[mat];
    float uFromChannel = geom.uBegin[mat] + (2 * id.channel() + 1 + fraction) * geom.halfChannelPitch[mat];
    if( id.die() ) uFromChannel += geom.dieGap[mat];
    uFromChannel += id.sipm() * geom.sipmPitch[mat];
    const float endPointX = geom.mirrorPointX[mat] + geom.ddxX[mat] * uFromChannel;
    const float endPointY = geom.mirrorPointY[mat] + geom.ddxY[mat] * uFromChannel;
    const float endPointZ = geom.mirrorPointZ[mat] + geom.ddxZ[mat] * uFromChannel;
    const float x0 = endPointX - dxdy * endPointY;
    const float z0 = endPointZ - dzdy * endPointY;

    // ORIGINAL: if(id.isBottom()) std::swap(yMin, yMax);
    float yMin = endPointY + id.isBottom() * globaldy;
    float yMax = endPointY + !id.isBottom() * globaldy;

    assert( pseudoSize < 9 && "Pseudosize of cluster is > 8. Out of range.");
    float werrX = invClusRes[pseudoSize];

    // Apparently the unique* methods are not designed to start at 0, therefore -16
    const uint32_t uniqueZone = ((id.uniqueQuarter() - 16) >> 1);
    uint32_t* hits_mat = hit_count.n_hits_mats + uniqueMat;
    uint32_t hitIndex = (*hits_mat)++;

    if(id.reversedZone())
      hitIndex = hit_count.mat_number_of_hits(uniqueMat) - 1 - hitIndex;

    assert( hitIndex < hit_count.mat_number_of_hits(uniqueMat));

    hitIndex += shared_mat_offsets[uniqueMat];

    hits.x0[hitIndex] = x0;
    hits.z0[hitIndex] = z0;
    hits.w[hitIndex] = werrX * werrX;
    hits.dxdy[hitIndex] = dxdy;
    hits.dzdy[hitIndex] = dzdy;
    hits.yMin[hitIndex] = yMin;
    hits.yMax[hitIndex] = yMax;
    hits.LHCbID[hitIndex] = lhcbid;
    hits.planeCode[hitIndex] = 2 * planeCode + (uniqueZone % 2); //  planeCode;
    hits.hitZone[hitIndex] = uniqueZone % 2;
  };

  // copied straight from FTRawBankDecoder.cpp
  auto make_clusters = [&](uint32_t firstChannel, uint16_t c, uint16_t c2, uint32_t uniqueMat) {
    unsigned int delta = (cell(c2) - cell(c));

    // fragmented clusters, size > 2*max size
    // only edges were saved, add middles now
    if (delta  > SciFiRawBankParams::clusterMaxWidth) {
      //add the first edge cluster, and then the middle clusters
      for(unsigned int i = SciFiRawBankParams::clusterMaxWidth; i < delta; i += SciFiRawBankParams::clusterMaxWidth){
        // all middle clusters will have same size as the first cluster,
        // so re-use the fraction
        make_cluster(firstChannel + i, fraction(c), 0, uniqueMat);
      }
      //add the last edge
      make_cluster  (firstChannel + delta, fraction(c2), 0, uniqueMat);
    } else { //big cluster size upto size 8
      unsigned int widthClus  =  2 * delta - 1 + fraction(c2);
      make_cluster(firstChannel+(widthClus-1)/2 - (SciFiRawBankParams::clusterMaxWidth - 1)/2,
                    (widthClus-1)%2, widthClus, uniqueMat);
    }//end if adjacent clusters
  };//End lambda make_clusters

  // Main execution loop
  for(uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x)
  {
    auto rawbank = event.getSciFiRawBank(i);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;
    if (*(last-1) == 0) --last;//Remove padding at the end
    for( ;  it < last; ++it ){ // loop over the clusters
      uint16_t c = *it;
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
      auto chid = SciFiChannelID(ch);
      uint32_t correctedMat = chid.correctedUniqueMat();

      if( !cSize(c) || it+1 == last ) { //No size flag or last cluster
        make_cluster(ch, fraction(c), 4, correctedMat);
      } else {//Flagged or not the last one.
        unsigned c2 = *(it+1);
        if( cSize(c2) && getLinkInBank(c) == getLinkInBank(c2) ) {
          make_clusters(ch,c,c2, correctedMat);
          ++it;
        } else {
          make_cluster(ch, fraction(c), 4, correctedMat);
        }
      }
    }
  }
}
