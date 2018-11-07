#include "UTCalculateNumberOfHits.cuh"

/**
 * @brief Calculates the number of hits to be decoded for the UT detector.
 */
__global__ void ut_calculate_number_of_hits (
  const uint32_t* dev_ut_raw_input,
  const uint32_t* dev_ut_raw_input_offsets,
  const char* ut_boards,
  const uint* dev_ut_region_offsets,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  uint32_t* dev_ut_hit_offsets
) {
  const uint32_t event_number = blockIdx.x;

  // Note: Once the lgenfe error is fixed in CUDA, we can switch to char* and drop the "/ sizeof(uint32_t);"
  const uint32_t event_offset = dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);
  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  uint32_t* hit_offsets = dev_ut_hit_offsets + event_number * number_of_unique_x_sectors;

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
  const UTBoards boards(ut_boards);

  for (uint32_t raw_bank_index = threadIdx.x; raw_bank_index < raw_event.number_of_raw_banks; raw_bank_index += blockDim.x) {
    const UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint32_t m_nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];

    for (uint32_t i = threadIdx.y; i < raw_bank.number_of_hits; i+=blockDim.y) {
      const uint32_t channelID = (raw_bank.data[i] & UTDecoding::chan_mask) >> UTDecoding::chan_offset;
      const uint32_t index = channelID / m_nStripsPerHybrid;
      const uint32_t fullChanIndex = raw_bank.sourceID * UTDecoding::ut_number_of_sectors_per_board + index;
      const uint32_t station       = boards.stations   [fullChanIndex] - 1;
      const uint32_t layer         = boards.layers     [fullChanIndex] - 1;
      const uint32_t detRegion     = boards.detRegions [fullChanIndex] - 1;
      const uint32_t sector        = boards.sectors    [fullChanIndex] - 1;

      // Calculate the index to get the geometry of the board
      const uint32_t idx = station * UTDecoding::ut_number_of_sectors_per_board + layer * 3 + detRegion;
      const uint32_t idx_offset = dev_ut_region_offsets[idx] + sector;

      uint* hits_sector_group = hit_offsets + dev_unique_x_sector_offsets[idx_offset];
      atomicAdd(hits_sector_group, 1);
    }
  }
}
