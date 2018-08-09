#include "UTDecoding.cuh"

#define leng_mask       0xFFFFU  // length
#define leng_offset           0  // length


#define frac_mask 0x0003U   // frac
#define chan_mask 0x3FFCU   // channel
#define thre_mask 0x8000U    // threshold

#define frac_offset      0  // frac
#define chan_offset      2  // channel
#define thre_offset     15  // threshold

__global__ void decode_raw_banks (
    uint32_t * dev_ut_raw_banks,
    uint32_t * dev_ut_raw_banks_offsets,
    char * ut_boards,
    char * ut_geometry,
    UTHits * dev_ut_hits_decoded,
    uint32_t dev_ut_number_of_raw_banks
) {
    uint32_t raw_bank_index = threadIdx.x;
    if (raw_bank_index >= dev_ut_number_of_raw_banks) return;

    const UTBoards boards(ut_boards);
    const UTGeometry geometry(ut_geometry);

    const uint32_t m_offset[12] = {0, 84, 164, 248, 332, 412, 496, 594, 674, 772, 870, 950};

    // WARNING: if "ut_max_number_of_hits_per_event" is not a multiple of "ut_number_of_layers"
    //          could cause hit overwrites
    const uint32_t chunkSize = (ut_max_number_of_hits_per_event / ut_number_of_layers);

    if (raw_bank_index == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < ut_number_of_layers; ++i) {
            dev_ut_hits_decoded->layer_offset[i] = i * chunkSize;
        }
        #pragma unroll
        for (uint32_t i = 0; i < ut_number_of_layers; ++i) {
            dev_ut_hits_decoded->n_hits_layers[i] = 0;
        }
    }

    const uint32_t offset = dev_ut_raw_banks_offsets[raw_bank_index];
    const uint16_t * raw_data = (uint16_t *)(dev_ut_raw_banks + offset + 1);

    const uint32_t sourceID = dev_ut_raw_banks[offset];
    const uint32_t hits = raw_data[0];


    const uint32_t m_nStripsPerHybrid = boards.stripsPerHybrids[sourceID];

    raw_data += 2;
    for (uint32_t i = 0; i < hits; ++i) {

        // Extract values from raw_data
        const uint16_t value     = raw_data[i];
        const uint32_t fracStrip = (value & frac_mask) >> frac_offset;
        const uint32_t channelID = (value & chan_mask) >> chan_offset;
        const uint32_t threshold = (value & thre_mask) >> thre_offset;

        // Calculate the relative index of the corresponding board
        const uint32_t index = channelID / m_nStripsPerHybrid;
        const uint32_t strip = channelID - (index * m_nStripsPerHybrid) + 1;

        const uint32_t fullChanIndex = sourceID * ut_number_of_sectors_per_board + index;
        const uint32_t station       = boards.stations   [fullChanIndex];
        const uint32_t layer         = boards.layers     [fullChanIndex];
        const uint32_t detRegion     = boards.detRegions [fullChanIndex];
        const uint32_t sector        = boards.sectors    [fullChanIndex];
        const uint32_t chanID        = boards.chanIDs    [fullChanIndex];

        // Calculate the index to get the geometry of the board
        const uint32_t idx = (station - 1) * ut_number_of_sectors_per_board + (layer - 1) * 3 + (detRegion - 1);
        const uint32_t idx_offset = m_offset[idx] + sector - 1;

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
        const uint32_t planeCode     = 2 * (station - 1 ) + (layer - 1 ) & 1;

        uint32_t hitIndex = atomicAdd(dev_ut_hits_decoded->n_hits_layers + planeCode, 1);

        // WARNING: if ("hitIndex" >= "chunkSize") there is a hit overwrite
        hitIndex += planeCode * chunkSize;
        dev_ut_hits_decoded->m_cos          [hitIndex] = cos;
        dev_ut_hits_decoded->m_yBegin       [hitIndex] = yBegin;
        dev_ut_hits_decoded->m_yEnd         [hitIndex] = yEnd;
        dev_ut_hits_decoded->m_zAtYEq0      [hitIndex] = zAtYEq0;
        dev_ut_hits_decoded->m_xAtYEq0      [hitIndex] = xAtYEq0;
        dev_ut_hits_decoded->m_weight       [hitIndex] = weight;
        dev_ut_hits_decoded->m_highThreshold[hitIndex] = highThreshold;
        dev_ut_hits_decoded->m_LHCbID       [hitIndex] = LHCbID;
        dev_ut_hits_decoded->m_planeCode    [hitIndex] = planeCode;
    }
}
