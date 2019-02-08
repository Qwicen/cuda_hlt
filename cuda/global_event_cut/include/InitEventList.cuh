#pragma once

#include "Common.h"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"

__global__ void init_event_list(uint* dev_event_list);

ALGORITHM(
  init_event_list,
  init_event_list_t,
  ARGUMENTS(
    dev_velo_raw_input,
    dev_velo_raw_input_offsets,
    dev_ut_raw_input,
    dev_ut_raw_input_offsets,
    dev_scifi_raw_input,
    dev_scifi_raw_input_offsets,
    dev_number_of_selected_events,
    dev_event_list))
