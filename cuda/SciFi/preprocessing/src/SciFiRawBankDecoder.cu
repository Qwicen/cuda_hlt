#include "SciFiRawBankDecoder.cuh"
#include "assert.h"

using namespace SciFi;

// Merge of PrStoreFTHit and RawBankDecoder.
__device__ void make_cluster (
  const int hit_index,
  const SciFiHitCount& hit_count,
  const SciFiGeometry& geom,
  uint32_t chan,
  uint8_t fraction,
  uint8_t pseudoSize,
  SciFiHits& hits)
{
  // TODO: Move to constants
  // maybe not hardcoded, or in another place
  // constexpr float invClusRes[] = {1/0.05, 1/0.08, 1/0.11, 1/0.14, 1/0.17, 1/0.20, 1/0.23, 1/0.26, 1/0.29};

  const SciFi::SciFiChannelID id {chan};

  // Offset to save space in geometry structure, see DumpFTGeometry.cpp
  const uint32_t mat = id.uniqueMat() - 512;
  // const uint32_t iQuarter = id.uniqueQuarter() - 16;
  const uint32_t planeCode = id.uniqueLayer() - 4;
  // See Kernel/LHCbID.h. Maybe no hardcoding?
  // const uint32_t lhcbid = (10u << 28) + chan;
  const float dxdy = geom.dxdy[mat];
  const float dzdy = geom.dzdy[mat];
  // const float globaldy = geom.globaldy[mat];
  float uFromChannel = geom.uBegin[mat] + (2 * id.channel() + 1 + fraction) * geom.halfChannelPitch[mat];
  if( id.die() ) uFromChannel += geom.dieGap[mat];
  uFromChannel += id.sipm() * geom.sipmPitch[mat];
  const float endPointX = geom.mirrorPointX[mat] + geom.ddxX[mat] * uFromChannel;
  const float endPointY = geom.mirrorPointY[mat] + geom.ddxY[mat] * uFromChannel;
  const float endPointZ = geom.mirrorPointZ[mat] + geom.ddxZ[mat] * uFromChannel;
  const float x0 = endPointX - dxdy * endPointY;
  const float z0 = endPointZ - dzdy * endPointY;

  // ORIGINAL: if(id.isBottom()) std::swap(yMin, yMax);
  // float yMin = endPointY + id.isBottom() * globaldy;
  // float yMax = endPointY + !id.isBottom() * globaldy;

  assert( pseudoSize < 9 && "Pseudosize of cluster is > 8. Out of range.");
  // float werrX = invClusRes[pseudoSize];

  // Apparently the unique* methods are not designed to start at 0, therefore -16
  const uint32_t uniqueZone = ((id.uniqueQuarter() - 16) >> 1);

  const uint plane_code = 2 * planeCode + (uniqueZone % 2);
  hits.x0[hit_index] = x0;
  hits.z0[hit_index] = z0;
  hits.channel[hit_index] = chan;
  hits.m_endPointY[hit_index] = endPointY;
  assert(fraction <= 0x1 && plane_code <= 0x1f && pseudoSize <= 0xf && mat <= 0x7ff);
  hits.assembled_datatype[hit_index] = fraction << 20 | plane_code << 15 | pseudoSize << 11 | mat;

  // TODO: Make accessors for these datatypes
  // hits.x0[hit_index] = x0;
  // hits.z0[hit_index] = z0;
  // hits.w[hit_index] = werrX * werrX;
  // hits.dxdy[hit_index] = dxdy;
  // hits.dzdy[hit_index] = dzdy;
  // hits.yMin[hit_index] = yMin;
  // hits.yMax[hit_index] = yMax;
  // hits.LHCbID[hit_index] = lhcbid;
  // hits.planeCode[hit_index] = 2 * planeCode + (uniqueZone % 2); //  planeCode;
  // hits.hitZone[hit_index] = uniqueZone % 2;
};

__global__ void scifi_raw_bank_decoder(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  uint *scifi_hits,
  char *scifi_geometry)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const SciFiGeometry geom {scifi_geometry};
  const auto event = SciFiRawEvent(scifi_events + scifi_event_offsets[event_number]);

  SciFiHits hits {scifi_hits, scifi_hit_count[number_of_events * SciFi::Constants::n_mats], &geom};
  SciFiHitCount hit_count;
  hit_count.typecast_after_prefix_sum(scifi_hit_count, event_number, number_of_events);
  const uint number_of_hits_in_event = hit_count.event_number_of_hits();

  for (int i=threadIdx.x; i<number_of_hits_in_event; i+=blockDim.x) {
    const uint32_t cluster_reference = hits.cluster_reference[hit_count.event_offset() + i];

    // Cluster reference:
    //   raw bank: 8 bits
    //   element (it): 8 bits
    //   Condition 1-2-3: 2 bits
    //   Condition 2.1-2.2: 1 bit
    //   Condition 2.1: log2(n+1) - 8 bits
    const int raw_bank_number = (cluster_reference >> 24) & 0xFF;
    const int it_number = (cluster_reference >> 16) & 0xFF;
    const int condition_1 = (cluster_reference >> 14) & 0x03;
    const int condition_2 = (cluster_reference >> 13) & 0x01;
    const int delta_parameter = cluster_reference & 0xFF;

    const auto rawbank = event.getSciFiRawBank(raw_bank_number);
    const uint16_t* it = rawbank.data + 2;
    it += it_number;

    const uint16_t c = *it;
    const uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
    const auto chid = SciFiChannelID(ch);
    //const uint32_t correctedMat = chid.correctedUniqueMat();

    // Call parameters for make_cluster
    uint32_t cluster_chan = ch;
    uint8_t cluster_fraction = fraction(c);
    uint8_t pseudoSize = 4;

    if (condition_1 == 0x01) {
      const auto c2 = *(it+1);
      const auto delta = cell(c2) - cell(c);

      if (condition_2 == 0x00) {
        pseudoSize = 0;

        if (delta_parameter == 0) {
          // add the last edge
          // make_cluster  (firstChannel + delta, fraction(c2), 0, uniqueMat);
          cluster_chan += delta;
          cluster_fraction = fraction(c2);
        } else {
          // make_cluster(firstChannel + i, fraction(c), 0, uniqueMat);
          cluster_chan += delta_parameter * SciFiRawBankParams::clusterMaxWidth;
        }
      } else { // (condition_2 == 0x01)
        // unsigned int widthClus  =  2 * delta - 1 + fraction(c2);
        // make_cluster(firstChannel+(widthClus-1)/2 - (SciFiRawBankParams::clusterMaxWidth - 1)/2,
        //               (widthClus-1)%2, widthClus, uniqueMat);
        const auto widthClus = 2*delta - 1 + fraction(c2);
        cluster_chan += (widthClus-1)/2 - (SciFiRawBankParams::clusterMaxWidth - 1)/2;
        cluster_fraction = (widthClus-1)%2;
        pseudoSize = widthClus;
      }
    }

    make_cluster(
      hit_count.event_offset() + i,
      hit_count,
      geom,
      cluster_chan,
      cluster_fraction,
      pseudoSize,
      hits);
  }
}
