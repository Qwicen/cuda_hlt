#pragma once

#include "Common.h"
#include "Handler.cuh"
#include "TrackBeamLineVertexFinder.h"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "patPV_Definitions.cuh"
#include <stdint.h>

__global__ void blpv_extrapolate(
  char* dev_kalmanvelo_states,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks);

ALGORITHM(blpv_extrapolate, blpv_extrapolate_t)
