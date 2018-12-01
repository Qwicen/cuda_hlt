#pragma once

#include "BeamlinePVConstants.cuh"
#include "Common.h"
#include "Handler.cuh"
#include "TrackBeamLineVertexFinder.h"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "patPV_Definitions.cuh"
#include <stdint.h>

__global__ void blpv_multi_fitter(
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks,
  PV::Vertex* dev_multi_fit_vertices,
  uint* dev_number_of_multi_fit_vertices);

ALGORITHM(blpv_multi_fitter, blpv_multi_fitter_t)
