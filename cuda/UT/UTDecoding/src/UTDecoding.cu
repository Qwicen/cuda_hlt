#include "UTDecoding.cuh"
#include <stdio.h>

#define leng_mask       0xFFFFU  // length
#define leng_offset           0  // length


#define frac_mask 0x0003U   // frac
#define chan_mask 0x3FFCU   // channel
#define thre_mask 0x8000U    // threshold

#define frac_offset      0  // frac
#define chan_offset      2  // channel
#define thre_offset     15  // threshold

__global__ void decode_raw_banks (
    uint32_t * dev_ut_raw_input,
    uint32_t * dev_ut_raw_input_offsets,
    char * ut_boards,
    char * ut_geometry,
    UTHits * dev_ut_hits_decoded,
    uint32_t dev_ut_number_of_raw_banks
) {
    const uint32_t tid          = threadIdx.x;
    const uint32_t event_number = blockIdx.x;
    const uint32_t event_offset = dev_ut_raw_input_offsets[event_number] / sizeof(uint32_t);

    const UTRawEvent raw_event(dev_ut_raw_input + event_offset);
    if (tid >= raw_event.number_of_raw_banks) return;

    const uint32_t m_offset[12] = {0, 84, 164, 248, 332, 412, 496, 594, 674, 772, 870, 950};
    const UTBoards boards(ut_boards);
    const UTGeometry geometry(ut_geometry);


    // WARNING: if "ut_max_number_of_hits_per_event" is not a multiple of "ut_number_of_layers"
    //          could cause hit overwrites
    const uint32_t chunkSize = (ut_max_number_of_hits_per_event / ut_number_of_layers);

    if (tid == 0) {

        #pragma unroll
        for (uint32_t i = 0; i < ut_number_of_layers; ++i) {
            dev_ut_hits_decoded[event_number].layer_offset[i] = i * chunkSize;
        }
        #pragma unroll
        for (uint32_t i = 0; i < ut_number_of_layers; ++i) {
            dev_ut_hits_decoded[event_number].n_hits_layers[i] = 0;
        }
    }

    __syncthreads();

    // number of raw_banks computed by a thread
//    const uint32_t raw_bank_chunk_size = (raw_event.number_of_raw_banks + blockDim.x - 1) / blockDim.x;
    
    for (uint32_t raw_bank_index = tid; raw_bank_index < raw_event.number_of_raw_banks; raw_bank_index += blockDim.x) {

        const uint32_t raw_bank_offset = raw_event.raw_bank_offsets[raw_bank_index];
        const UTRawBank raw_bank(raw_event.data + raw_bank_offset);
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

            // WARNING: if ("hitIndex" >= "chunkSize") there is a hit overwrite
            hitIndex += planeCode * chunkSize;
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
