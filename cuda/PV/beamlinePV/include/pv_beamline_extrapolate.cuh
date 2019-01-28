#pragma once

#include "Common.h"
#include "Handler.cuh"
#include "TrackBeamLineVertexFinder.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "patPV_Definitions.cuh"
#include <stdint.h>

__global__ void pv_beamline_extrapolate(
  char* dev_velo_kalman_beamline_states,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks);

ALGORITHM(pv_beamline_extrapolate, pv_beamline_extrapolate_t)
