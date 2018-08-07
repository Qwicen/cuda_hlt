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
    uint32_t * dev_ut_stripsPerHybrid,
    UTExpandedChannelIDs * dev_ut_expanded_channels,
    UTGeometry * dev_ut_geometry,
    UTHits * dev_ut_hits_decoded,
    uint32_t dev_ut_number_of_raw_banks
) {

    const uint32_t m_offset[12] = {0, 84, 164, 248, 332, 412, 496, 594, 674, 772, 870, 950};

    // WARNING: if "ut_max_number_of_hits_per_event" is not a multiple of "ut_number_of_layers"
    //          could cause hit overwrites
    const uint32_t chunkSize = (ut_max_number_of_hits_per_event / ut_number_of_layers);

	uint32_t raw_bank_index = threadIdx.x;
    if (raw_bank_index >= dev_ut_number_of_raw_banks) return;


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


    const uint32_t m_nStripsPerHybrid = dev_ut_stripsPerHybrid[sourceID];

    raw_data += 2;
    for (uint32_t i = 0; i < hits; ++i) {

        const uint16_t value = raw_data[i];
        const uint32_t fracStrip = (value & frac_mask) >> frac_offset;
        const uint32_t channelID = (value & chan_mask) >> chan_offset;
        const uint32_t threshold = (value & thre_mask) >> thre_offset;

        const uint32_t index = channelID / m_nStripsPerHybrid;
        const uint32_t strip = channelID - (index * m_nStripsPerHybrid) + 1; // Add one because offline strips start at one.

        const uint32_t fullChanIndex = sourceID * 6 + index;
        const uint32_t station   = dev_ut_expanded_channels->stations   [fullChanIndex];
        const uint32_t layer     = dev_ut_expanded_channels->layers     [fullChanIndex];
        const uint32_t detRegion = dev_ut_expanded_channels->detRegions [fullChanIndex];
        const uint32_t sector    = dev_ut_expanded_channels->sectors    [fullChanIndex];
        const uint32_t chanID    = dev_ut_expanded_channels->chanIDs    [fullChanIndex];

        // get index offset corresponding to the station/layer/region we want
        const uint32_t idx = (station - 1) * 6 + (layer - 1) * 3 + (detRegion - 1);
        const uint32_t idx_offset = m_offset[idx] + sector - 1;

        uint32_t m_firstStrip   = dev_ut_geometry->m_firstStrip [idx_offset];
        const float m_pitch     = dev_ut_geometry->m_pitch      [idx_offset];
        // const float m_dxdy      = dev_ut_geometry->m_dxdy       [idx_offset];
        // const float m_dzdy      = dev_ut_geometry->m_dzdy       [idx_offset];
        const float m_dy        = dev_ut_geometry->m_dy         [idx_offset];
        const float m_dp0diX    = dev_ut_geometry->m_dp0diX     [idx_offset];
        const float m_dp0diY    = dev_ut_geometry->m_dp0diY     [idx_offset];
        const float m_dp0diZ    = dev_ut_geometry->m_dp0diZ     [idx_offset];
        const float m_p0X       = dev_ut_geometry->m_p0X        [idx_offset];
        const float m_p0Y       = dev_ut_geometry->m_p0Y        [idx_offset];
        const float m_p0Z       = dev_ut_geometry->m_p0Z        [idx_offset];
        const float m_cosAngle  = dev_ut_geometry->m_cosAngle   [idx_offset];


        const float numstrips = (fracStrip / 4.f) + strip - m_firstStrip;

        const float ut_cos              = m_cosAngle;
        const float ut_yBegin           = m_p0Y + numstrips * m_dp0diY;
        const float ut_yEnd             = m_dy  + ut_yBegin;
        // const float ut_dxDy             = m_dxdy;
        const float ut_zAtYEq0          = m_p0Z + numstrips * m_dp0diZ;
        const float ut_xAtYEq0          = m_p0X + numstrips * m_dp0diX;
        const float ut_weight           = 1.f / (m_pitch / sqrtf( 12.f ));
        const uint32_t ut_highThreshold = threshold;
        const uint32_t ut_LHCbID        = chanID + strip;
        const uint32_t ut_planeCode = 2 * (station - 1 ) + (layer - 1 ) & 1;

        uint32_t hitIndex = atomicAdd(dev_ut_hits_decoded->n_hits_layers + ut_planeCode, 1);

        // WARNING: if ("hitIndex" >= "chunkSize") there is a hit overwrite
        hitIndex += ut_planeCode * chunkSize;
        dev_ut_hits_decoded->m_cos          [hitIndex] = ut_cos;
        dev_ut_hits_decoded->m_yBegin       [hitIndex] = ut_yBegin;
        dev_ut_hits_decoded->m_yEnd         [hitIndex] = ut_yEnd;
        dev_ut_hits_decoded->m_zAtYEq0      [hitIndex] = ut_zAtYEq0;
        dev_ut_hits_decoded->m_xAtYEq0      [hitIndex] = ut_xAtYEq0;
        dev_ut_hits_decoded->m_weight       [hitIndex] = ut_weight;
        dev_ut_hits_decoded->m_highThreshold[hitIndex] = ut_highThreshold;
        dev_ut_hits_decoded->m_LHCbID       [hitIndex] = ut_LHCbID;
        dev_ut_hits_decoded->m_planeCode    [hitIndex] = ut_planeCode;
    }
}
