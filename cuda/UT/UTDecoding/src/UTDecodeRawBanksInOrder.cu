#include "UTDecodeRawBanksInOrder.cuh"

__global__ void ut_decode_raw_banks_in_order(
  const uint32_t *dev_ut_raw_input,
  const uint32_t *dev_ut_raw_input_offsets,
  const char *ut_boards, const char *ut_geometry,
  const uint *dev_ut_region_offsets,
  const uint *dev_unique_x_sector_layer_offsets,
  const uint *dev_unique_x_sector_offsets,
  const uint32_t *dev_ut_hit_offsets,
  uint32_t *dev_ut_hits,
  uint32_t *dev_ut_hit_count,
  uint* dev_hit_permutations)
{
  const uint32_t number_of_events = gridDim.x;
  const uint32_t event_number = blockIdx.x;
  const uint layer_number = blockIdx.y;
  const uint32_t event_offset =
      dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[VeloUTTracking::n_layers];

  const UTHitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  UTHits ut_hits {dev_ut_hits, dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]};

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
  const UTBoards boards(ut_boards);
  const UTGeometry geometry(ut_geometry);

  // if (threadIdx.x==0) {
  //   printf("%i, %i\n", event_hit_starting_offset, ut_hit_offsets.event_number_of_hits());
  // }

  const uint layer_offset = ut_hit_offsets.layer_offset(layer_number);
  const uint layer_number_of_hits = ut_hit_offsets.layer_number_of_hits(layer_number);

  for (int i=threadIdx.x; i<layer_number_of_hits; i+=blockDim.x) {
    const uint hit_index = layer_offset + i;
    const uint32_t raw_bank_hit_index = ut_hits.raw_bank_index[dev_hit_permutations[hit_index]];
    const uint raw_bank_index = raw_bank_hit_index >> 24;
    const uint hit_index_inside_raw_bank = raw_bank_hit_index & 0xFFFFFF;

    const UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint16_t value = raw_bank.data[hit_index_inside_raw_bank];
    const uint32_t nStripsPerHybrid =
        boards.stripsPerHybrids[raw_bank.sourceID];

    // Extract values from raw_data
    const uint32_t fracStrip = (value & UTDecoding::frac_mask) >> UTDecoding::frac_offset;
    const uint32_t channelID = (value & UTDecoding::chan_mask) >> UTDecoding::chan_offset;
    const uint32_t threshold = (value & UTDecoding::thre_mask) >> UTDecoding::thre_offset;

    // Calculate the relative index of the corresponding board
    const uint32_t index = channelID / nStripsPerHybrid;
    const uint32_t strip = channelID - (index * nStripsPerHybrid) + 1;

    const uint32_t fullChanIndex =
        raw_bank.sourceID * UTDecoding::ut_number_of_sectors_per_board + index;
    const uint32_t station = boards.stations[fullChanIndex] - 1;
    const uint32_t layer = boards.layers[fullChanIndex] - 1;
    const uint32_t detRegion = boards.detRegions[fullChanIndex] - 1;
    const uint32_t sector = boards.sectors[fullChanIndex] - 1;
    const uint32_t chanID = boards.chanIDs[fullChanIndex];

    // Calculate the index to get the geometry of the board
    const uint32_t idx =
        station * UTDecoding::ut_number_of_sectors_per_board + layer * 3 + detRegion;
    const uint32_t idx_offset = dev_ut_region_offsets[idx] + sector;

    const uint32_t m_firstStrip = geometry.firstStrip[idx_offset];
    const float m_pitch = geometry.pitch[idx_offset];
    const float m_dy = geometry.dy[idx_offset];
    const float m_dp0diX = geometry.dp0diX[idx_offset];
    const float m_dp0diY = geometry.dp0diY[idx_offset];
    const float m_dp0diZ = geometry.dp0diZ[idx_offset];
    const float m_p0X = geometry.p0X[idx_offset];
    const float m_p0Y = geometry.p0Y[idx_offset];
    const float m_p0Z = geometry.p0Z[idx_offset];

    const float numstrips = (fracStrip / 4.f) + strip - m_firstStrip;

    // Calculate values of the hit
    const float yBegin = m_p0Y + numstrips * m_dp0diY;
    const float yEnd = m_dy + yBegin;
    const float zAtYEq0 = m_p0Z + numstrips * m_dp0diZ;
    const float xAtYEq0 = m_p0X + numstrips * m_dp0diX;
    const float weight = 1.f / (m_pitch / sqrtf(12.f));
    const uint32_t highThreshold = threshold;
    const uint32_t LHCbID = chanID + strip;
    const uint32_t planeCode = 2 * station + (layer & 1);

    ut_hits.yBegin[hit_index] = yBegin;
    ut_hits.yEnd[hit_index] = yEnd;
    ut_hits.zAtYEq0[hit_index] = zAtYEq0;
    ut_hits.xAtYEq0[hit_index] = xAtYEq0;
    ut_hits.weight[hit_index] = weight;
    ut_hits.highThreshold[hit_index] = highThreshold;
    ut_hits.LHCbID[hit_index] = LHCbID;
    ut_hits.planeCode[hit_index] = planeCode;
  }
}
