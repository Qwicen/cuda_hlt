#include "SciFiRawBankDecoderV4.cuh"
#include "assert.h"

using namespace SciFi;

// Merge of PrStoreFTHit and RawBankDecoder.
__device__ void make_cluster_v4(
  const int hit_index,
  const SciFiGeometry& geom,
  uint32_t chan,
  uint8_t fraction,
  uint8_t pseudoSize,
  SciFi::Hits& hits)
{
  const SciFi::SciFiChannelID id {chan};

  // Offset to save space in geometry structure, see DumpFTGeometry.cpp
  const uint32_t mat = id.uniqueMat() - 512;
  const uint32_t planeCode = id.uniqueLayer() - 4;
  const float dxdy = geom.dxdy[mat];
  const float dzdy = geom.dzdy[mat];
  float uFromChannel = geom.uBegin[mat] + (2 * id.channel() + 1 + fraction) * geom.halfChannelPitch[mat];
  if (id.die()) uFromChannel += geom.dieGap[mat];
  uFromChannel += id.sipm() * geom.sipmPitch[mat];
  const float endPointX = geom.mirrorPointX[mat] + geom.ddxX[mat] * uFromChannel;
  const float endPointY = geom.mirrorPointY[mat] + geom.ddxY[mat] * uFromChannel;
  const float endPointZ = geom.mirrorPointZ[mat] + geom.ddxZ[mat] * uFromChannel;
  const float x0 = endPointX - dxdy * endPointY;
  const float z0 = endPointZ - dzdy * endPointY;

  assert(pseudoSize < 9 && "Pseudosize of cluster is > 8. Out of range.");

  // Apparently the unique* methods are not designed to start at 0, therefore -16
  const uint32_t uniqueZone = ((id.uniqueQuarter() - 16) >> 1);

  const uint plane_code = 2 * planeCode + (uniqueZone % 2);
  hits.x0[hit_index] = x0;
  hits.z0[hit_index] = z0;
  hits.channel[hit_index] = chan;
  hits.m_endPointY[hit_index] = endPointY;
  assert(fraction <= 0x1 && plane_code <= 0x1f && pseudoSize <= 0xf && mat <= 0x7ff);
  hits.assembled_datatype[hit_index] = fraction << 20 | plane_code << 15 | pseudoSize << 11 | mat;
};

__global__ void scifi_raw_bank_decoder_v4(
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

  const SciFiGeometry geom {scifi_geometry};
  const auto event = SciFiRawEvent(scifi_events + scifi_event_offsets[selected_event_number]);

  SciFi::Hits hits {
    scifi_hits, scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats], &geom, dev_inv_clus_res};
  const SciFi::HitCount hit_count {scifi_hit_count, event_number};
  const uint number_of_hits_in_last_zones = hit_count.number_of_hits_in_zones_without_mat_groups();

  for (int i = threadIdx.x; i < number_of_hits_in_last_zones; i += blockDim.x) {
    const uint32_t cluster_reference = hits.cluster_reference[hit_count.offset_zones_without_mat_groups() + i];

    const int raw_bank_number = (cluster_reference >> 8) & 0xFF;
    const int it_number = (cluster_reference) &0xFF;

    const auto rawbank = event.getSciFiRawBank(raw_bank_number);
    const uint16_t* it = rawbank.data + 2;
    it += it_number;

    const uint16_t c = *it;
    const uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
    const auto chid = SciFiChannelID(ch);

    // Call parameters for make_cluster
    uint32_t cluster_chan = ch;
    uint8_t cluster_fraction = fraction(c);
    uint8_t pseudoSize = cSize(c) ? 0 : 4;

    make_cluster_v4(hit_count.offset_zones_without_mat_groups() + i, geom, cluster_chan, cluster_fraction, pseudoSize, hits);
  }
}
