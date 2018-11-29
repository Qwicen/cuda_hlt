#pragma once

#include "VeloDefinitions.cuh"
#include "patPV_Definitions.cuh"
#include <stdint.h>
#include "VeloEventModel.cuh"
#include "Common.h"
#include "Handler.cuh"
#include "VeloConsolidated.cuh"
#include "TrackBeamLineVertexFinder.h"
#include "BeamlinePVConstants.cuh"


    

  


__global__ void blpv_peak(
  float* dev_zhisto,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks);




 ALGORITHM(blpv_peak, blpv_peak_t)