#pragma once

#include <stdint.h>

__global__ void decode_raw_banks (
    uint32_t * dev_ut_raw_banks,
    uint32_t * dev_ut_raw_banks_offsets,
    uint32_t * dev_ut_sourceIDs,
    uint32_t * dev_ut_number_of_hits,
    uint32_t dev_ut_number_of_raw_banks
);
