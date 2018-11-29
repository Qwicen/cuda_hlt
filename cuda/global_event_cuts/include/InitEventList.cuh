#pragma once

#include "Common.h"
#include "Handler.cuh" 

__global__ void init_event_list( uint* dev_event_list ); 
 
ALGORITHM(init_event_list, init_event_list_t)
