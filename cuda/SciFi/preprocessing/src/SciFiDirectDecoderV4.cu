#include "SciFiDirectDecoderV4.cuh"
#include "assert.h"

using namespace SciFi;

__global__ void scifi_direct_decoder_v4(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  uint *scifi_hits,
  char* scifi_geometry,
  const float* dev_inv_clus_res)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const SciFiGeometry geom(scifi_geometry);
  const auto event = SciFiRawEvent(scifi_events + scifi_event_offsets[event_number]);

  SciFi::Hits hits {scifi_hits, scifi_hit_count[number_of_events * SciFi::Constants::n_mats], &geom, dev_inv_clus_res};
  SciFi::HitCount hit_count;
  hit_count.typecast_after_prefix_sum(scifi_hit_count, event_number, number_of_events);

  for (uint i_raw_bank = threadIdx.x; i_raw_bank < SciFi::Constants::n_consecutive_raw_banks; i_raw_bank += blockDim.x) {
    const uint raw_bank_offset = hit_count.mat_group_offset(i_raw_bank);

    const uint j = (i_raw_bank / 10) % 4;
    const bool reverse_cluster_order = j == 1 | j == 2;

    const uint k = i_raw_bank % 10;
    const bool reverse_raw_bank_order = k < 5;
    const uint current_raw_bank = reverse_raw_bank_order ?
      5 * (i_raw_bank / 5) + (4 - i_raw_bank % 5) :
      i_raw_bank;

    const auto rawbank = event.getSciFiRawBank(current_raw_bank);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    if (*(last - 1) == 0) --last; // Remove padding at the end
    const uint number_of_clusters = last - it;

    for (int i_cluster = threadIdx.y; i_cluster < number_of_clusters; i_cluster += blockDim.y) {
      const uint16_t current_cluster = reverse_cluster_order ? (number_of_clusters - 1 - i_cluster) : i_cluster;

      uint16_t c = *(it + current_cluster);
      uint8_t cluster_fraction = fraction(c);
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
      const SciFi::SciFiChannelID id {ch};

      // Offset to save space in geometry structure, see DumpFTGeometry.cpp
      const uint32_t mat = id.uniqueMat() - 512;
      const uint32_t planeCode = id.uniqueLayer() - 4;
      const float dxdy = geom.dxdy[mat];
      const float dzdy = geom.dzdy[mat];
      float uFromChannel = geom.uBegin[mat] + (2 * id.channel() + 1 + cluster_fraction) * geom.halfChannelPitch[mat];
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
      const uint hit_index = raw_bank_offset + current_cluster;
      const uint8_t pseudoSize = cSize(c) ? 0 : 4;

      hits.x0[hit_index] = x0;
      hits.z0[hit_index] = z0;
      hits.channel[hit_index] = ch;
      hits.m_endPointY[hit_index] = endPointY;
      assert(fraction <= 0x1 && plane_code <= 0x1f && pseudoSize <= 0xf && mat <= 0x7ff);
      hits.assembled_datatype[hit_index] = cluster_fraction << 20 | plane_code << 15 | pseudoSize << 11 | mat;
    }
  }
}
