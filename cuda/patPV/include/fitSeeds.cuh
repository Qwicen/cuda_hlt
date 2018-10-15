#include "patPV_Definitions.cuh"

__global__ void fitSeeds(
    Vertex* dev_vertex,
  int * dev_number_vertex,
  XYZPoint * dev_seeds,
  uint * dev_number_seeds);