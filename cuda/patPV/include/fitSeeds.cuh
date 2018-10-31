#include "patPV_Definitions.cuh"
#include "VeloEventModel.cuh"

__global__ void fitSeeds(
    Vertex* dev_vertex,
  int * dev_number_vertex,
  XYZPoint * dev_seeds,
  uint * dev_number_seeds,
  Velo::State* dev_velo_states,
  int * dev_atomics_storage);

__device__ bool fitVertex( XYZPoint& seedPoint,
              Velo::State * host_velo_states,
             Vertex& vtx,
              int number_of_tracks) ;

__device__ double getTukeyWeight(double trchi2, int iter) ;