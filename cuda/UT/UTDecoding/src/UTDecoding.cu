#include "UTDecoding.cuh"

__global__ void decode_raw_banks (
        uint32_t * dev_ut_raw_banks,
        uint32_t * dev_ut_raw_banks_offsets,
        uint32_t * dev_ut_sourceIDs,
        uint32_t * dev_ut_number_of_hits,
        uint32_t dev_ut_number_of_raw_banks
) {
	int raw_bank_index = threadIdx.x;

    if (raw_bank_index >= dev_ut_number_of_raw_banks) return;

    uint32_t offset = dev_ut_raw_banks_offsets[raw_bank_index];

    const uint32_t sourceID = dev_ut_raw_banks[offset];
    const uint32_t hits = ( dev_ut_raw_banks[offset + 1] & 0x0000FFFFU );

    dev_ut_sourceIDs[raw_bank_index] = sourceID; 
    dev_ut_number_of_hits[raw_bank_index] = hits;
}
