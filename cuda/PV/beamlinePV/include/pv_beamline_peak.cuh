#pragma once

#include "BeamlinePVConstants.cuh"
#include "Common.h"
#include "Handler.cuh"
#include "ArgumentsPV.cuh"
#include "TrackBeamLineVertexFinder.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "patPV_Definitions.cuh"
#include "FloatOperations.cuh"
#include <cstdint>

__global__ void pv_beamline_peak(
  float* dev_zhisto,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks,
  uint number_of_events);

ALGORITHM(pv_beamline_peak, pv_beamline_peak_t,
  ARGUMENTS(dev_zhisto,
    dev_zpeaks,
    dev_number_of_zpeaks))
