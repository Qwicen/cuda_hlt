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


    

  


__global__ void blpv_histo(int * dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks,
  float* dev_zhisto);




 ALGORITHM(blpv_histo, blpv_histo_t)