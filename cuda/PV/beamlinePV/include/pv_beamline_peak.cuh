#pragma once

#include "BeamlinePVConstants.cuh"
#include "Common.h"
#include "Handler.cuh"
#include "TrackBeamLineVertexFinder.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "patPV_Definitions.cuh"
#include <stdint.h>

__global__ void pv_beamline_peak(
  float* dev_zhisto,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks,
  uint number_of_events);

ALGORITHM(pv_beamline_peak, pv_beamline_peak_t)
