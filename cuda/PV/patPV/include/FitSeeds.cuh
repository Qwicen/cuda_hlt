#pragma once

#include <stdint.h>
#include "Common.h"
#include "Handler.cuh"
#include "ArgumentsVelo.cuh"
#include "ArgumentsPV.cuh"
#include "patPV_Definitions.cuh"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"
#include "PV_Definitions.cuh"

__global__ void fit_seeds(
  PatPV::Vertex* dev_vertex,
  int * dev_number_vertex,
  PatPV::XYZPoint * dev_seeds,
  uint * dev_number_seeds,
  char* dev_velo_kalman_beamline_states,
  int * dev_atomics_storage,
  uint* dev_velo_track_hit_number);

__device__ bool fit_vertex(PatPV::XYZPoint& seedPoint,
              Velo::Consolidated::States velo_states,
              PV::Vertex& vtx,
              int number_of_tracks,
              uint tracks_offset) ;

__device__ float get_tukey_weight(float trchi2, int iter) ;

ALGORITHM(fit_seeds, pv_fit_seeds_t,
  ARGUMENTS(
    dev_vertex,
    dev_number_vertex,
    dev_seeds,
    dev_number_seeds,
    dev_velo_kalman_beamline_states,
    dev_atomics_velo,
    dev_velo_track_hit_number
))
