#include "UTPreDecode.cuh"

/**
 * Iterate over raw banks / hits and store only the Y coordinate,
 * and an uint32_t encoding the following:
 * raw_bank number and hit id inside the raw bank.
 * Let's refer to this array as raw_bank_hits.
 */
__global__ void ut_pre_decode(
  const char* dev_ut_raw_input,
  const uint32_t* dev_ut_raw_input_offsets,
  const uint* dev_event_list,
  const char* ut_boards,
  const char* ut_geometry,
  const uint* dev_ut_region_offsets,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  const uint32_t* dev_ut_hit_offsets,
  uint32_t* dev_ut_hits,
  uint32_t* dev_ut_hit_count)
{
  const uint32_t number_of_events = gridDim.x;
  const uint32_t event_number = blockIdx.x;
  const uint selected_event_number = dev_event_list[event_number];

  const uint32_t event_offset = dev_ut_raw_input_offsets[selected_event_number];

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint32_t* hit_offsets = dev_ut_hit_offsets + event_number * number_of_unique_x_sectors;
  uint32_t* hit_count = dev_ut_hit_count + event_number * number_of_unique_x_sectors;

  UT::Hits ut_hits {dev_ut_hits, dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]};

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
  const UTBoards boards(ut_boards);
  const UTGeometry geometry(ut_geometry);

  for (uint32_t raw_bank_index = threadIdx.x; raw_bank_index < raw_event.number_of_raw_banks;
       raw_bank_index += blockDim.x) {
    const UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint32_t m_nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];

    for (uint32_t i = threadIdx.y; i < raw_bank.number_of_hits; i += blockDim.y) {
      // Extract values from raw_data
      const uint16_t value = raw_bank.data[i];
      const uint32_t fracStrip = (value & UT::Decoding::frac_mask) >> UT::Decoding::frac_offset;
      const uint32_t channelID = (value & UT::Decoding::chan_mask) >> UT::Decoding::chan_offset;

      // Calculate the relative index of the corresponding board
      const uint32_t index = channelID / m_nStripsPerHybrid;
      const uint32_t strip = channelID - (index * m_nStripsPerHybrid) + 1;

      const uint32_t fullChanIndex = raw_bank.sourceID * UT::Decoding::ut_number_of_sectors_per_board + index;
      const uint32_t station = boards.stations[fullChanIndex] - 1;
      const uint32_t layer = boards.layers[fullChanIndex] - 1;
      const uint32_t detRegion = boards.detRegions[fullChanIndex] - 1;
      const uint32_t sector = boards.sectors[fullChanIndex] - 1;

      // Calculate the index to get the geometry of the board
      const uint32_t idx = station * UT::Decoding::ut_number_of_sectors_per_board + layer * 3 + detRegion;
      const uint32_t idx_offset = dev_ut_region_offsets[idx] + sector;

      const uint32_t firstStrip = geometry.firstStrip[idx_offset];
      const float dp0diX = geometry.dp0diX[idx_offset];
      const float dp0diY = geometry.dp0diY[idx_offset];
      const float p0Y = geometry.p0Y[idx_offset];

      const float numstrips = 0.25f * fracStrip + strip - firstStrip;

      // Make a composed value made out of:
      // (first 16 bits of yBegin) | (first 16 bits of xAtYEq0_local)
      //
      // Rationale:
      // Sorting in floats is done the same way as for ints,
      // the bigger the binary number, the bigger the float (it's a designed property
      // of the float format). Also, the format of a float is as follows:
      // * 1 bit: sign
      // * 8 bits: exponent
      // * 23 bits: mantissa
      // By using the first 16 bits of each, we get the sign, exponent and 7 bits
      // of the mantissa, for both Y and X, which is enough to account for the
      // cases where yBegin was repeated.
      const float yBegin = p0Y + numstrips * dp0diY;
      const float xAtYEq0_local = numstrips * dp0diX;
      const int* yBegin_p = reinterpret_cast<const int*>(&yBegin);
      const int* xAtYEq0_local_p = reinterpret_cast<const int*>(&xAtYEq0_local);

      // The second value needs to be changed its sign using the 2's complement logic (operator-),
      // if the signs of both values differ.
      const short composed_0 = (yBegin_p[0] & 0xFFFF0000) >> 16;
      short composed_1 = (xAtYEq0_local_p[0] & 0xFFFF0000) >> 16;
      const bool sign_0 = composed_0 & 0x8000;
      const bool sign_1 = composed_1 & 0x8000;
      if (sign_0 ^ sign_1) {
        composed_1 = -composed_1;
      }

      const int composed_value = (composed_0 << 16) & 0xFFFF0000 | composed_1 & 0x0000FFFF;
      const float* composed_value_float = reinterpret_cast<const float*>(&composed_value);

      const uint base_sector_group_offset = dev_unique_x_sector_offsets[idx_offset];
      uint* hits_count_sector_group = hit_count + base_sector_group_offset;

      const uint current_hit_count = atomicAdd(hits_count_sector_group, 1);
      assert(current_hit_count < hit_offsets[base_sector_group_offset + 1] - hit_offsets[base_sector_group_offset]);

      const uint hit_index = hit_offsets[base_sector_group_offset] + current_hit_count;
      ut_hits.yBegin[hit_index] = composed_value_float[0];

      // Raw bank hit index:
      // [raw bank 8 bits] [hit id inside raw bank 24 bits]
      assert(i < (0x1 << 24));
      const uint32_t raw_bank_hit_index = raw_bank_index << 24 | i;
      ut_hits.raw_bank_index[hit_index] = raw_bank_hit_index;
    }
  }
}
