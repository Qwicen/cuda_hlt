#pragma once

#include "../../velo/common/include/VeloDefinitions.cuh"
#include "patPV_Definitions.cuh"
#include "VeloEventModel.cuh"
//#include "AdaptivePV3DFitter.cuh"
//#include "PVSeedTool.cuh"


//






__global__ void patPV(
    Velo::State* dev_velo_states,
  int * dev_atomics_storage,
  PatPV::Vertex * dev_outvtxvec,
  uint * dev_number_of_vertex
);