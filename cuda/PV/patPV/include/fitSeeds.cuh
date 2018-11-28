#pragma once

#include <stdint.h>
#include "Common.h"
#include "Handler.cuh"
#include "patPV_Definitions.cuh"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"

__global__ void fit_seeds(
  PatPV::Vertex* dev_vertex,
  int * dev_number_vertex,
  PatPV::XYZPoint * dev_seeds,
  uint * dev_number_seeds,
  char* dev_kalmanvelo_states,
  int * dev_atomics_storage,
  uint* dev_velo_track_hit_number);

__device__ bool fit_vertex(PatPV::XYZPoint& seedPoint,
              Velo::Consolidated::States velo_states,
              PatPV::Vertex& vtx,
              int number_of_tracks,
              uint tracks_offset) ;

__device__ PatPV::myfloat get_tukey_weight(PatPV::myfloat trchi2, int iter) ;

ALGORITHM(fit_seeds, pv_fit_seeds_t)
