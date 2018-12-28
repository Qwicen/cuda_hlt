#pragma once

#include "VeloDefinitions.cuh"
#include "patPV_Definitions.cuh"
#include <stdint.h>
#include "VeloEventModel.cuh"
#include "Common.h"
#include "Handler.cuh"
#include "VeloConsolidated.cuh"


__global__ void get_seeds(
  char* dev_velo_kalman_beamline_states,
  int * dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PatPV::XYZPoint * dev_seeds,
  uint * dev_number_seeds
);

 __device__ int find_clusters(PatPV::vtxCluster * vclus, float * zclusters, int number_of_clusters);


 ALGORITHM(get_seeds, pv_get_seeds_t)
