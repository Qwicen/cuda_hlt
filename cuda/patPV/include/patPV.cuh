
#include "../../velo/common/include/VeloDefinitions.cuh"
#include "patPV_Definitions.cuh"
//#include "AdaptivePV3DFitter.cuh"
//#include "PVSeedTool.cuh"


//






__global__ void patPV(
    VeloState* dev_velo_states,
  int * dev_atomics_storage,
  Vertex * dev_outvtxvec,
  uint * dev_number_of_vertex
);