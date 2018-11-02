#include <stdint.h>
#include "Common.h"
#include "Handler.cuh"
#include "patPV_Definitions.cuh"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"

__global__ void fitSeeds(
  Vertex* dev_vertex,
  int * dev_number_vertex,
  XYZPoint * dev_seeds,
  uint * dev_number_seeds,
  uint* dev_kalmanvelo_states,
  int * dev_atomics_storage,
  uint* dev_velo_track_hit_number);

__device__ bool fitVertex( XYZPoint& seedPoint,
              Velo::Consolidated::States velo_states,
             Vertex& vtx,
              int number_of_tracks,
              uint tracks_offset) ;

__device__ double getTukeyWeight(double trchi2, int iter) ;

 ALGORITHM(fitSeeds, fitSeeds_t)