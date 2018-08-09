#pragma once

#include "UTDefinitions.cuh"

__global__ void decode_raw_banks (
    uint32_t * dev_ut_raw_banks,
    uint32_t * dev_ut_raw_banks_offsets,
    char * ut_boards,
    char * ut_geometry,
    UTHits * dev_ut_hits_decoded,
    uint32_t dev_ut_number_of_raw_banks
);
