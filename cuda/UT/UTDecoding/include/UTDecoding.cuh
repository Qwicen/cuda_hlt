#pragma once

#include "UTDefinitions.cuh"

__global__ void ut_estimate_number_of_hits (
    const uint32_t * __restrict__ dev_ut_raw_input,
    const uint32_t * __restrict__ dev_ut_raw_input_offsets,
    const char * __restrict__ ut_boards,
    UTHits * __restrict__ dev_ut_hits_decoded
);

__global__ void decode_raw_banks (
    const uint32_t * __restrict__ dev_ut_raw_input,
    const uint32_t * __restrict__ dev_ut_raw_input_offsets,
    const char * __restrict__ ut_boards,
    const char * __restrict__ ut_geometry,
    UTHits * __restrict__ dev_ut_hits_decoded
);

