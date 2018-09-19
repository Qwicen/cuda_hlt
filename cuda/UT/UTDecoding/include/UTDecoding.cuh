#pragma once

#include "UTDefinitions.cuh"
#include "VeloUTDefinitions.cuh"

__global__ void ut_calculate_number_of_hits (
  const uint32_t* dev_ut_raw_input,
  const uint32_t* dev_ut_raw_input_offsets,
  const char* ut_boards,
  const uint* dev_ut_region_offsets,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  uint32_t* dev_ut_hit_offsets
);

__global__ void decode_raw_banks (
  const uint32_t* dev_ut_raw_input,
  const uint32_t* dev_ut_raw_input_offsets,
  const char* ut_boards,
  const char* ut_geometry,
  const uint* dev_ut_region_offsets,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  const uint32_t* dev_ut_hit_offsets,
  uint32_t* dev_ut_hits_decoded,
  uint32_t* dev_ut_hit_count
);

