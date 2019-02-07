#pragma once

#include "Common.h"
#include "Handler.cuh" 
#include "SciFiRaw.cuh"
#include "UTRaw.cuh"
#include "ArgumentsCommon.cuh"

static constexpr uint maxSciFiUTClusters = 9750; // check tha this removes 10% of the events!

__global__ void global_event_cut(
  char* dev_ut_raw_input,
  uint* dev_ut_raw_input_offsets,
  char* dev_scifi_raw_input,
  uint* dev_scifi_raw_input_offsets,  
  uint* number_of_selected_events,
  uint* event_list
); 

ALGORITHM(global_event_cut, global_event_cut_t,
  ARGUMENTS(
    dev_ut_raw_input,
    dev_ut_raw_input_offsets,
    dev_scifi_raw_input,
    dev_scifi_raw_input_offsets,
    dev_number_of_selected_events,
    dev_event_list
))
