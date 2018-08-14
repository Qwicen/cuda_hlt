#pragma once

#include "UTDefinitions.cuh"

__global__ void decode_raw_banks (
    uint32_t * dev_ut_raw_input,
    uint32_t * dev_ut_raw_input_offsets,
    char * ut_boards,
    char * ut_geometry,
    UTHits * dev_ut_hits_decoded
);
