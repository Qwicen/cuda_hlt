#include "InitEventList.cuh"

__global__ void init_event_list( uint* dev_event_list) {
  
  const uint event_number = threadIdx.x;
  
  dev_event_list[event_number] = event_number;
}
 
