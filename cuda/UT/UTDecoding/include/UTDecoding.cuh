#pragma once

#include <stdint.h>
#include "UTDefinitions.cuh"

__global__ void decode_raw_banks (
    uint32_t * dev_ut_raw_banks,
    uint32_t * dev_ut_raw_banks_offsets,
    uint32_t * dev_ut_stripsPerHybrid,
    UTExpandedChannelIDs * dev_ut_expanded_channels,
    UTGeometry * dev_ut_geometry,
    UTHits * dev_ut_hits_decoded,
    uint32_t dev_ut_number_of_raw_banks
);
