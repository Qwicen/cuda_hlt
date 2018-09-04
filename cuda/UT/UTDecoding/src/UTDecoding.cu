#include "UTDecoding.cuh"
#include <stdio.h>

#define frac_mask 0x0003U   // frac
#define chan_mask 0x3FFCU   // channel
#define thre_mask 0x8000U    // threshold

#define frac_offset      0  // frac
#define chan_offset      2  // channel
#define thre_offset     15  // threshold

// HINT: This estimation could be done inside the decoding kernel just for the first hit


__global__ void ut_estimate_number_of_hits (
  const uint32_t * __restrict__ dev_ut_raw_input,
  const uint32_t * __restrict__ dev_ut_raw_input_offsets,
  const char * __restrict__ ut_boards,
  UTHits * __restrict__ dev_ut_hits_decoded
) {
  const uint32_t event_number = blockIdx.x;
  const uint32_t event_offset = dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);

  // Initialize hit layers to 0
  for (int i=threadIdx.x; i<ut_number_of_layers; i+=blockDim.x) {
    dev_ut_hits_decoded[event_number].n_hits_layers[i] = 0;
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

    uint32_t * hits_layer = dev_ut_hits_decoded[event_number].n_hits_layers + planeCode;
    uint32_t hitIndex = atomicAdd(hits_layer, raw_bank.number_of_hits);
  }

  __syncthreads();

  // TODO: This is a prefix sum
  if (threadIdx.x == 0) {
    uint32_t layer_offset = 0;
    dev_ut_hits_decoded[event_number].layer_offset[0] = 0;
    
    // Note: Does the #pragma affect here?
    // #pragma unroll
    for (uint32_t i = 1; i < ut_number_of_layers; ++i) {
      const uint32_t n_hits_layers = dev_ut_hits_decoded[event_number].n_hits_layers[i-1];
      layer_offset += n_hits_layers;
      dev_ut_hits_decoded[event_number].layer_offset[i] = layer_offset;
    }
  }
}

__global__ void decode_raw_banks (
  const uint32_t * __restrict__ dev_ut_raw_input,
  const uint32_t * __restrict__ dev_ut_raw_input_offsets,
  const char * __restrict__ ut_boards,
  const char * __restrict__ ut_geometry,
  UTHits * __restrict__ dev_ut_hits_decoded
) {
  const uint32_t tid          = threadIdx.x;
  const uint32_t event_number = blockIdx.x;
  const uint32_t event_offset = dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);

  __shared__ uint32_t layer_offset[ut_number_of_layers];

  for (int i=threadIdx.x; i<ut_number_of_layers; i+=blockDim.x) {
    layer_offset[i] = dev_ut_hits_decoded[event_number].layer_offset[i];
  }

  for (int i=threadIdx.x; i<ut_number_of_layers; i+=blockDim.x) {
    dev_ut_hits_decoded[event_number].n_hits_layers[i] = 0;
  }

  const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
  if (tid >= raw_event.number_of_raw_banks) return;

  const uint32_t m_offset[12] = {0, 84, 164, 248, 332, 412, 496, 594, 674, 772, 870, 950};
  const UTBoards boards(ut_boards);
  const UTGeometry geometry(ut_geometry);

  // uint32_t layer_offset[ut_number_of_layers];
  // #pragma unroll
  // for (uint32_t i = 0; i < ut_number_of_layers; ++i) {
  //     layer_offset[i] = dev_ut_hits_decoded[event_number].layer_offset[i];
  // }

  // Due to layer_offset and n_hits_layers initialization
  __syncthreads();

  for (uint32_t raw_bank_index = tid; raw_bank_index < raw_event.number_of_raw_banks; raw_bank_index += blockDim.x) {
    UTRawBank raw_bank = raw_event.getUTRawBank(raw_bank_index);
    const uint32_t m_nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];

    for (uint32_t i = 0; i < raw_bank.number_of_hits; ++i) {
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

      uint32_t * hits_layer = dev_ut_hits_decoded[event_number].n_hits_layers + planeCode;
      uint32_t hitIndex = atomicAdd(hits_layer, 1);

      hitIndex += layer_offset[planeCode];
      dev_ut_hits_decoded[event_number].m_cos          [hitIndex] = cos;
      dev_ut_hits_decoded[event_number].m_yBegin       [hitIndex] = yBegin;
      dev_ut_hits_decoded[event_number].m_yEnd         [hitIndex] = yEnd;
      dev_ut_hits_decoded[event_number].m_zAtYEq0      [hitIndex] = zAtYEq0;
      dev_ut_hits_decoded[event_number].m_xAtYEq0      [hitIndex] = xAtYEq0;
      dev_ut_hits_decoded[event_number].m_weight       [hitIndex] = weight;
      dev_ut_hits_decoded[event_number].m_highThreshold[hitIndex] = highThreshold;
      dev_ut_hits_decoded[event_number].m_LHCbID       [hitIndex] = LHCbID;
      dev_ut_hits_decoded[event_number].m_planeCode    [hitIndex] = planeCode;
    }
  }
}


