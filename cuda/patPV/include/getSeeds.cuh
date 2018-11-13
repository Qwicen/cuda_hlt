#pragma once

#include "../../velo/common/include/VeloDefinitions.cuh"
#include "patPV_Definitions.cuh"
#include <stdint.h>
#include "VeloEventModel.cuh"
#include "Common.h"
#include "Handler.cuh"
#include "VeloConsolidated.cuh"


__global__ void getSeeds(
  uint* dev_kalmanvelo_states,
  int * dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PatPV::XYZPoint * dev_seeds,
  uint * dev_number_seeds);

 __device__ int findClusters(PatPV::vtxCluster * vclus, double * zclusters, int number_of_clusters);


 ALGORITHM(getSeeds, getSeeds_t)