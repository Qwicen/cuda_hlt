#pragma once

#include "UTDefinitions.cuh"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsUT.cuh"

__global__ void ut_calculate_number_of_hits(
  const char* dev_ut_raw_input,
  const uint32_t* dev_ut_raw_input_offsets,
  const char* ut_boards,
  const uint* dev_ut_region_offsets,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  uint32_t* dev_ut_hit_offsets,
  const uint* dev_event_list);

ALGORITHM(
  ut_calculate_number_of_hits,
  ut_calculate_number_of_hits_t,
  ARGUMENTS(dev_ut_raw_input, dev_ut_raw_input_offsets, dev_ut_hit_offsets, dev_event_list))
