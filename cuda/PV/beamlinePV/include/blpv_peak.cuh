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

__global__ void blpv_peak(
  float* dev_zhisto,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks,
  uint number_of_events);

ALGORITHM(blpv_peak, blpv_peak_t)
