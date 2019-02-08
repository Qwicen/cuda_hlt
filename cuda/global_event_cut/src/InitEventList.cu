#include "InitEventList.cuh"
/**
   This algorithm ensures that we can run w/o applying
   global event cuts: the event_list is trivially filled
   here instead of in global_event_cuts
 */
__global__ void init_event_list(uint* dev_event_list)
{
  const auto event_number = blockIdx.x * blockDim.x + threadIdx.x;
  dev_event_list[event_number] = event_number;
}
