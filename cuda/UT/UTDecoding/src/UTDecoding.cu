#include "UTDecoding.cuh"
#include <stdio.h>

#define frac_mask 0x0003U // frac
#define chan_mask 0x3FFCU // channel
#define thre_mask 0x8000U // threshold

#define frac_offset 0  // frac
#define chan_offset 2  // channel
#define thre_offset 15 // threshold

/**
 * @brief Calculates the number of hits to be decoded for the UT detector.
 */
__global__ void ut_calculate_number_of_hits(
  const uint32_t *dev_ut_raw_input, const uint32_t *dev_ut_raw_input_offsets,
  const char *ut_boards, const uint *dev_ut_region_offsets,
  const uint *dev_unique_x_sector_layer_offsets,
  const uint *dev_unique_x_sector_offsets, uint32_t *dev_ut_hit_offsets)
{
  const uint32_t event_number = blockIdx.x;

  // Note: Once the lgenfe error is fixed in CUDA, we can switch to char* and
  // drop the "/ sizeof(uint32_t);"
  const uint32_t event_offset =
      dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);
  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  uint32_t *hit_offsets =
      dev_ut_hit_offsets + event_number * number_of_unique_x_sectors;

  // Initialize hit layers to 0
  for (int i = threadIdx.x; i < number_of_unique_x_sectors; i += blockDim.x) {
    hit_offsets[i] = 0;
  }

  __syncthreads();

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
  const UTBoards boards(ut_boards);

  for (uint32_t raw_bank_index = threadIdx.x;
       raw_bank_index < raw_event.number_of_raw_banks;
       raw_bank_index += blockDim.x) {
    const UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint32_t m_nStripsPerHybrid =
        boards.stripsPerHybrids[raw_bank.sourceID];

    for (uint32_t i = threadIdx.y; i < raw_bank.number_of_hits;
         i += blockDim.y) {
      const uint32_t channelID = (raw_bank.data[i] & chan_mask) >> chan_offset;
      const uint32_t index = channelID / m_nStripsPerHybrid;
      const uint32_t fullChanIndex =
          raw_bank.sourceID * ut_number_of_sectors_per_board + index;
      const uint32_t station = boards.stations[fullChanIndex] - 1;
      const uint32_t layer = boards.layers[fullChanIndex] - 1;
      const uint32_t detRegion = boards.detRegions[fullChanIndex] - 1;
      const uint32_t sector = boards.sectors[fullChanIndex] - 1;

      // Calculate the index to get the geometry of the board
      const uint32_t idx =
          station * ut_number_of_sectors_per_board + layer * 3 + detRegion;
      const uint32_t idx_offset = dev_ut_region_offsets[idx] + sector;

      uint *hits_sector_group =
          hit_offsets + dev_unique_x_sector_offsets[idx_offset];
      atomicAdd(hits_sector_group, 1);
    }
  }
}

/**
 * Iterate over raw banks / hits and store only the Y coordinate,
 * and an uint32_t encoding the following:
 * raw_bank number and hit id inside the raw bank.
 * Let's refer to this array as raw_bank_hits.
 */
__global__ void ut_pre_decode(
  const uint32_t *dev_ut_raw_input,
  const uint32_t *dev_ut_raw_input_offsets,
  const char *ut_boards, const char *ut_geometry,
  const uint *dev_ut_region_offsets,
  const uint *dev_unique_x_sector_layer_offsets,
  const uint *dev_unique_x_sector_offsets,
  const uint32_t *dev_ut_hit_offsets,
  uint32_t *dev_ut_hits,
  uint32_t *dev_ut_hit_count)
{
  const uint32_t number_of_events = gridDim.x;
  const uint32_t event_number = blockIdx.x;
  const uint32_t event_offset =
      dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint32_t *hit_offsets =
      dev_ut_hit_offsets + event_number * number_of_unique_x_sectors;
  uint32_t *hit_count =
      dev_ut_hit_count + event_number * number_of_unique_x_sectors;

  UTHits ut_hits;
  ut_hits.typecast_unsorted(
      dev_ut_hits,
      dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]);

  if (threadIdx.y == 0) {
    for (int i = threadIdx.x; i < number_of_unique_x_sectors; i += blockDim.x) {
      hit_count[i] = 0;
    }
  }

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
  const UTBoards boards(ut_boards);
  const UTGeometry geometry(ut_geometry);

  // Due to layer_offset and n_hits_layers initialization
  __syncthreads();

  for (uint32_t raw_bank_index = threadIdx.x;
       raw_bank_index < raw_event.number_of_raw_banks;
       raw_bank_index += blockDim.x) {
    UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint32_t m_nStripsPerHybrid =
        boards.stripsPerHybrids[raw_bank.sourceID];

    for (uint32_t i = threadIdx.y; i < raw_bank.number_of_hits;
         i += blockDim.y) {
      // Extract values from raw_data
      const uint16_t value = raw_bank.data[i];
      const uint32_t fracStrip = (value & frac_mask) >> frac_offset;
      const uint32_t channelID = (value & chan_mask) >> chan_offset;

      // Calculate the relative index of the corresponding board
      const uint32_t index = channelID / m_nStripsPerHybrid;
      const uint32_t strip = channelID - (index * m_nStripsPerHybrid) + 1;

      const uint32_t fullChanIndex =
          raw_bank.sourceID * ut_number_of_sectors_per_board + index;
      const uint32_t station = boards.stations[fullChanIndex] - 1;
      const uint32_t layer = boards.layers[fullChanIndex] - 1;
      const uint32_t detRegion = boards.detRegions[fullChanIndex] - 1;
      const uint32_t sector = boards.sectors[fullChanIndex] - 1;

      // Calculate the index to get the geometry of the board
      const uint32_t idx =
          station * ut_number_of_sectors_per_board + layer * 3 + detRegion;
      const uint32_t idx_offset = dev_ut_region_offsets[idx] + sector;

      const uint32_t firstStrip = geometry.firstStrip[idx_offset];
      const float dp0diY = geometry.dp0diY[idx_offset];
      const float p0Y = geometry.p0Y[idx_offset];

      const float numstrips = (fracStrip / 4.f) + strip - firstStrip;

      // Calculate just Y value
      const float yBegin = p0Y + numstrips * dp0diY;

      const uint base_sector_group_offset =
          dev_unique_x_sector_offsets[idx_offset];
      uint *hits_count_sector_group = hit_count + base_sector_group_offset;

      const uint current_hit_count = atomicAdd(hits_count_sector_group, 1);
      assert(current_hit_count < hit_offsets[base_sector_group_offset + 1] -
                                     hit_offsets[base_sector_group_offset]);

      const uint hit_index =
          hit_offsets[base_sector_group_offset] + current_hit_count;
      ut_hits.yBegin[hit_index] = yBegin;

      // Raw bank hit index:
      // [raw bank 8 bits] [hit id inside raw bank 24 bits]
      assert(i < (0x1 << 24));
      const uint32_t raw_bank_hit_index = raw_bank_index << 24 | i;
      ut_hits.raw_bank_index[hit_index] = raw_bank_hit_index;
    }
  }
}

__global__ void ut_decode_raw_banks_in_order(
  const uint32_t *dev_ut_raw_input,
  const uint32_t *dev_ut_raw_input_offsets,
  const char *ut_boards, const char *ut_geometry,
  const uint *dev_ut_region_offsets,
  const uint *dev_unique_x_sector_layer_offsets,
  const uint *dev_unique_x_sector_offsets,
  const uint32_t *dev_ut_hit_offsets,
  uint32_t *dev_ut_hits,
  uint32_t *dev_ut_hit_count)
{
  const uint32_t number_of_events = gridDim.x;
  const uint32_t event_number = blockIdx.x;
  const uint32_t event_offset =
      dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint32_t *hit_offsets =
      dev_ut_hit_offsets + event_number * number_of_unique_x_sectors;
  uint32_t *hit_count =
      dev_ut_hit_count + event_number * number_of_unique_x_sectors;

  UTHits ut_hits;
  ut_hits.typecast_sorted(
      dev_ut_hits,
      dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]);

  const UTHitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
  const UTBoards boards(ut_boards);
  const UTGeometry geometry(ut_geometry);

  // As a first implementation, iterate over all hits and store coalesced
  const uint event_hit_starting_offset = ut_hit_offsets.event_offset();

  // if (threadIdx.x==0) {
  //   printf("%i, %i\n", event_hit_starting_offset, ut_hit_offsets.event_number_of_hits());
  // }

  for (int i=threadIdx.x; i<ut_hit_offsets.event_number_of_hits(); i+=blockDim.x) {
    const uint hit_index = event_hit_starting_offset + i;
    const uint32_t raw_bank_hit_index = ut_hits.raw_bank_index[hit_index];
    const uint raw_bank_index = raw_bank_hit_index >> 24;
    const uint hit_index_inside_raw_bank = raw_bank_hit_index & 0xFFFFFF;

    const UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint16_t value = raw_bank.data[hit_index_inside_raw_bank];
    const uint32_t nStripsPerHybrid =
        boards.stripsPerHybrids[raw_bank.sourceID];

    // Extract values from raw_data
    const uint32_t fracStrip = (value & frac_mask) >> frac_offset;
    const uint32_t channelID = (value & chan_mask) >> chan_offset;
    const uint32_t threshold = (value & thre_mask) >> thre_offset;

    // Calculate the relative index of the corresponding board
    const uint32_t index = channelID / nStripsPerHybrid;
    const uint32_t strip = channelID - (index * nStripsPerHybrid) + 1;

    const uint32_t fullChanIndex =
        raw_bank.sourceID * ut_number_of_sectors_per_board + index;
    const uint32_t station = boards.stations[fullChanIndex] - 1;
    const uint32_t layer = boards.layers[fullChanIndex] - 1;
    const uint32_t detRegion = boards.detRegions[fullChanIndex] - 1;
    const uint32_t sector = boards.sectors[fullChanIndex] - 1;
    const uint32_t chanID = boards.chanIDs[fullChanIndex];

    // Calculate the index to get the geometry of the board
    const uint32_t idx =
        station * ut_number_of_sectors_per_board + layer * 3 + detRegion;
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

    ut_hits.yEnd[hit_index] = yEnd;
    ut_hits.zAtYEq0[hit_index] = zAtYEq0;
    ut_hits.xAtYEq0[hit_index] = xAtYEq0;
    ut_hits.weight[hit_index] = weight;
    ut_hits.highThreshold[hit_index] = highThreshold;
    ut_hits.LHCbID[hit_index] = LHCbID;
    ut_hits.planeCode[hit_index] = planeCode;
  }
}

__global__ void decode_raw_banks(const uint32_t *dev_ut_raw_input,
                                 const uint32_t *dev_ut_raw_input_offsets,
                                 const char *ut_boards, const char *ut_geometry,
                                 const uint *dev_ut_region_offsets,
                                 const uint *dev_unique_x_sector_layer_offsets,
                                 const uint *dev_unique_x_sector_offsets,
                                 const uint32_t *dev_ut_hit_offsets,
                                 uint32_t *dev_ut_hits,
                                 uint32_t *dev_ut_hit_count) {
  const uint32_t number_of_events = gridDim.x;
  const uint32_t event_number = blockIdx.x;
  const uint32_t event_offset =
      dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint32_t *hit_offsets =
      dev_ut_hit_offsets + event_number * number_of_unique_x_sectors;
  uint32_t *hit_count =
      dev_ut_hit_count + event_number * number_of_unique_x_sectors;

  UTHits ut_hits;
  ut_hits.typecast_unsorted(
      dev_ut_hits,
      dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]);

  if (threadIdx.y == 0) {
    for (int i = threadIdx.x; i < number_of_unique_x_sectors; i += blockDim.x) {
      hit_count[i] = 0;
    }
  }

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
  const UTBoards boards(ut_boards);
  const UTGeometry geometry(ut_geometry);

  // Due to layer_offset and n_hits_layers initialization
  __syncthreads();

  for (uint32_t raw_bank_index = threadIdx.x;
       raw_bank_index < raw_event.number_of_raw_banks;
       raw_bank_index += blockDim.x) {
    UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint32_t m_nStripsPerHybrid =
        boards.stripsPerHybrids[raw_bank.sourceID];

    for (uint32_t i = threadIdx.y; i < raw_bank.number_of_hits;
         i += blockDim.y) {
      // Extract values from raw_data
      const uint16_t value = raw_bank.data[i];
      const uint32_t fracStrip = (value & frac_mask) >> frac_offset;
      const uint32_t channelID = (value & chan_mask) >> chan_offset;
      const uint32_t threshold = (value & thre_mask) >> thre_offset;

      // Calculate the relative index of the corresponding board
      const uint32_t index = channelID / m_nStripsPerHybrid;
      const uint32_t strip = channelID - (index * m_nStripsPerHybrid) + 1;

      const uint32_t fullChanIndex =
          raw_bank.sourceID * ut_number_of_sectors_per_board + index;
      const uint32_t station = boards.stations[fullChanIndex] - 1;
      const uint32_t layer = boards.layers[fullChanIndex] - 1;
      const uint32_t detRegion = boards.detRegions[fullChanIndex] - 1;
      const uint32_t sector = boards.sectors[fullChanIndex] - 1;
      const uint32_t chanID = boards.chanIDs[fullChanIndex];

      // Calculate the index to get the geometry of the board
      const uint32_t idx =
          station * ut_number_of_sectors_per_board + layer * 3 + detRegion;
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

      const uint base_sector_group_offset =
          dev_unique_x_sector_offsets[idx_offset];
      uint *hits_count_sector_group = hit_count + base_sector_group_offset;

      const uint current_hit_count = atomicAdd(hits_count_sector_group, 1);
      assert(current_hit_count < hit_offsets[base_sector_group_offset + 1] -
                                     hit_offsets[base_sector_group_offset]);

      const uint hit_index =
          hit_offsets[base_sector_group_offset] + current_hit_count;
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
}
