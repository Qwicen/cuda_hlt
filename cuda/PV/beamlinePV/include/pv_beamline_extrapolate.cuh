#pragma once

#include "Common.h"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsVelo.cuh"
#include "ArgumentsPV.cuh"
#include "TrackBeamLineVertexFinder.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "patPV_Definitions.cuh"
#include "FloatOperations.cuh"
#include <cstdint>

__global__ void pv_beamline_extrapolate(
  char* dev_velo_kalman_beamline_states,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks);

ALGORITHM(pv_beamline_extrapolate, pv_beamline_extrapolate_t,
  ARGUMENTS(
    dev_velo_kalman_beamline_states,
    dev_atomics_velo,
    dev_velo_track_hit_number,
    dev_pvtracks
))
