#pragma once

#include "UTDefinitions.cuh"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsUT.cuh"
#include "UTEventModel.cuh"

__global__ void ut_decode_raw_banks_in_order(
  const char *dev_ut_raw_input,
  const uint32_t *dev_ut_raw_input_offsets,
  const uint *dev_event_list,
  const char *ut_boards, const char *ut_geometry,
  const uint *dev_ut_region_offsets,
  const uint *dev_unique_x_sector_layer_offsets,
  const uint *dev_unique_x_sector_offsets,
  const uint32_t *dev_ut_hit_offsets,
  uint32_t *dev_ut_hits,
  uint32_t *dev_ut_hit_count,
  uint* dev_hit_permutations);

ALGORITHM(ut_decode_raw_banks_in_order, ut_decode_raw_banks_in_order_t,
  ARGUMENTS(dev_ut_raw_input,
    dev_ut_raw_input_offsets,
    dev_ut_hits,
    dev_ut_hit_offsets,
    dev_ut_hit_count,
    dev_ut_hit_permutations,
    dev_event_list))
