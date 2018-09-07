#include "UTDecoding.cuh"
#include <stdio.h>

#define frac_mask 0x0003U   // frac
#define chan_mask 0x3FFCU   // channel
#define thre_mask 0x8000U    // threshold

#define frac_offset      0  // frac
#define chan_offset      2  // channel
#define thre_offset     15  // threshold

/**
 * @brief Calculates the number of hits to be decoded for the UT detector.
 */
__global__ void ut_calculate_number_of_hits (
  const uint32_t* __restrict__ dev_ut_raw_input,
  const uint32_t* __restrict__ dev_ut_raw_input_offsets,
  const char* __restrict__ ut_boards,
  uint32_t* __restrict__ dev_ut_hit_count
) {
  const uint32_t number_of_events = gridDim.x;
  const uint32_t event_number = blockIdx.x;

  // Note: Once the lgenfe error is fixed in CUDA, we can switch to char* and drop the "/ sizeof(uint32_t);"
  const uint32_t event_offset = dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);

  // Note: Only in this algorithm, n_hit_layers is accessed by the shift event_number * VeloUTTracking::n_layers.
  // After this, we will do a prefix sum, so instead we will have the offset at the same location.
  // For follow up algorithms, the recommended way to access this datatype is (note the +1, stemming from the prefix sum):
  //  uint32_t* layer_offset = dev_ut_hit_count + event_number * VeloUTTracking::n_layers;
  //  uint32_t* n_hit_layers = dev_ut_hit_count + number_of_events * VeloUTTracking::n_layers + 1 + event_number * VeloUTTracking::n_layers;
  uint32_t* n_hits_layers = dev_ut_hit_count + event_number * VeloUTTracking::n_layers;

  // Initialize hit layers to 0
  for (int i=threadIdx.x; i<VeloUTTracking::n_layers; i+=blockDim.x) {
    n_hits_layers[i] = 0;
  }

  __syncthreads();

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
  const UTBoards boards(ut_boards);

  for (uint32_t raw_bank_index = threadIdx.x; raw_bank_index < raw_event.number_of_raw_banks; raw_bank_index += blockDim.x) {
    const UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint32_t m_nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];
    const uint32_t channelID = (raw_bank.data[0] & chan_mask) >> chan_offset;
    const uint32_t index = channelID / m_nStripsPerHybrid;
    const uint32_t fullChanIndex = raw_bank.sourceID * ut_number_of_sectors_per_board + index;
    const uint32_t station       = boards.stations   [fullChanIndex] - 1;
    const uint32_t layer         = boards.layers     [fullChanIndex] - 1;
    const uint32_t planeCode     = 2 * station + (layer & 1);

    uint32_t* hits_layer = n_hits_layers + planeCode;
    atomicAdd(hits_layer, raw_bank.number_of_hits);
  }
}

__global__ void decode_raw_banks (
  const uint32_t* __restrict__ dev_ut_raw_input,
  const uint32_t* __restrict__ dev_ut_raw_input_offsets,
  const char* __restrict__ ut_boards,
  const char* __restrict__ ut_geometry,
  uint32_t* __restrict__ dev_ut_hits,
  uint32_t* __restrict__ dev_ut_hit_count
) {
  const uint32_t number_of_events = gridDim.x;
  const uint32_t event_number = blockIdx.x;
  const uint32_t event_offset = dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);
  const uint32_t* layer_offset = dev_ut_hit_count + event_number * VeloUTTracking::n_layers;
  uint32_t* n_hits_layers = dev_ut_hit_count + number_of_events * VeloUTTracking::n_layers + 1 + event_number * VeloUTTracking::n_layers;
  
  UTHits ut_hits;
  ut_hits.typecast_unsorted(dev_ut_hits, dev_ut_hit_count[number_of_events * VeloUTTracking::n_layers]);

  __shared__ uint32_t shared_layer_offset[ut_number_of_layers];

  for (int i=threadIdx.x; i<ut_number_of_layers; i+=blockDim.x) {
    shared_layer_offset[i] = layer_offset[i];
  }

  for (int i=threadIdx.x; i<ut_number_of_layers; i+=blockDim.x) {
    n_hits_layers[i] = 0;
  }

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);

  const uint32_t m_offset[12] = {0, 84, 164, 248, 332, 412, 496, 594, 674, 772, 870, 950};
  const UTBoards boards(ut_boards);
  const UTGeometry geometry(ut_geometry);

  // Due to layer_offset and n_hits_layers initialization
  __syncthreads();

  for (uint32_t raw_bank_index = threadIdx.x; raw_bank_index < raw_event.number_of_raw_banks; raw_bank_index += blockDim.x) {
    UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint32_t m_nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];

    for (uint32_t i = threadIdx.y; i < raw_bank.number_of_hits; i+=blockDim.y) {
      // Extract values from raw_data
      const uint16_t value     = raw_bank.data[i];
      const uint32_t fracStrip = (value & frac_mask) >> frac_offset;
      const uint32_t channelID = (value & chan_mask) >> chan_offset;
      const uint32_t threshold = (value & thre_mask) >> thre_offset;

      // Calculate the relative index of the corresponding board
      const uint32_t index = channelID / m_nStripsPerHybrid;
      const uint32_t strip = channelID - (index * m_nStripsPerHybrid) + 1;

      const uint32_t fullChanIndex = raw_bank.sourceID * ut_number_of_sectors_per_board + index;
      const uint32_t station       = boards.stations   [fullChanIndex] - 1;
      const uint32_t layer         = boards.layers     [fullChanIndex] - 1;
      const uint32_t detRegion     = boards.detRegions [fullChanIndex] - 1;
      const uint32_t sector        = boards.sectors    [fullChanIndex] - 1;
      const uint32_t chanID        = boards.chanIDs    [fullChanIndex];

      // Calculate the index to get the geometry of the board
      const uint32_t idx = station * ut_number_of_sectors_per_board + layer * 3 + detRegion;
      const uint32_t idx_offset = m_offset[idx] + sector;

      const uint32_t m_firstStrip = geometry.firstStrip [idx_offset];
      const float    m_pitch      = geometry.pitch      [idx_offset];
      const float    m_dy         = geometry.dy         [idx_offset];
      const float    m_dp0diX     = geometry.dp0diX     [idx_offset];
      const float    m_dp0diY     = geometry.dp0diY     [idx_offset];
      const float    m_dp0diZ     = geometry.dp0diZ     [idx_offset];
      const float    m_p0X        = geometry.p0X        [idx_offset];
      const float    m_p0Y        = geometry.p0Y        [idx_offset];
      const float    m_p0Z        = geometry.p0Z        [idx_offset];
      const float    m_cosAngle   = geometry.cos        [idx_offset];

      const float numstrips = (fracStrip / 4.f) + strip - m_firstStrip;

      // Calculate values of the hit
      const float    cos           = m_cosAngle;
      const float    yBegin        = m_p0Y + numstrips * m_dp0diY;
      const float    yEnd          = m_dy  + yBegin;
      const float    zAtYEq0       = m_p0Z + numstrips * m_dp0diZ;
      const float    xAtYEq0       = m_p0X + numstrips * m_dp0diX;
      const float    weight        = 1.f / (m_pitch / sqrtf( 12.f ));
      const uint32_t highThreshold = threshold;
      const uint32_t LHCbID        = chanID + strip;
      const uint32_t planeCode     = 2 * station + (layer & 1);

      uint32_t* hits_layer = n_hits_layers + planeCode;
      uint32_t hitIndex = atomicAdd(hits_layer, 1);

      hitIndex += shared_layer_offset[planeCode];
      ut_hits.m_cos[hitIndex]           = cos;
      ut_hits.m_yBegin[hitIndex]        = yBegin;
      ut_hits.m_yEnd[hitIndex]          = yEnd;
      ut_hits.m_zAtYEq0[hitIndex]       = zAtYEq0;
      ut_hits.m_xAtYEq0[hitIndex]       = xAtYEq0;
      ut_hits.m_weight[hitIndex]        = weight;
      ut_hits.m_highThreshold[hitIndex] = highThreshold;
      ut_hits.m_LHCbID[hitIndex]        = LHCbID;
      ut_hits.m_planeCode[hitIndex]     = planeCode;
    }
  }
}
