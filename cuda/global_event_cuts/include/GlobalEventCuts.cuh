#pragma once

#include "Common.h"
#include "Handler.cuh" 
#include "SciFiRaw.cuh"

static constexpr uint maxSciFiUTClusters = 11000; // CHANGE

__global__ void global_event_cuts(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  char* dev_ut_raw_input,
  uint* dev_ut_raw_input_offsets,
  char* dev_scifi_raw_input,
  uint* dev_scifi_raw_input_offsets  
); 

ALGORITHM(global_event_cuts, global_event_cuts_t)
