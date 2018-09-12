#include "EstimateClusterCount.cuh"
#include <stdio.h>
#include "assert.h"

using namespace FT;

__global__ void estimate_cluster_count(uint *ft_event_offsets, uint *dev_ft_cluster_offsets, uint *dev_ft_cluster_num, char *ft_events) {
  // TODO: Optimize parallelization.
  //printf("Preprocessing FT event %u, offset: %u...\n", blockIdx.x, ft_event_offsets[blockIdx.x]);
  const uint event_id = blockIdx.x;

  //if first thread...?
  *(dev_ft_cluster_offsets + event_id) = 0;
  *(dev_ft_cluster_num) = 0;
  __syncthreads();

  const auto event = FTRawEvent(ft_events + ft_event_offsets[event_id]);
  const uint rawbank_chunk = (event.number_of_raw_banks + blockDim.x - 1) / blockDim.x; // ceiling int division

  assert(event.version == 5u);
  uint byte_count = 0;
  for(uint rawbank = threadIdx.x; rawbank < event.number_of_raw_banks; rawbank+=event.number_of_raw_banks/rawbank_chunk)
  {
    byte_count += event.raw_bank_offset[rawbank] - (rawbank == 0? 0 : event.raw_bank_offset[rawbank-1]);
  }
  //approx. 2 bytes per cluster in v5 (overestimates a little). See LHCb::FTDAQ::nbFTClusters (FTDAQHelper.cpp)
  atomicAdd(dev_ft_cluster_offsets + event_id, byte_count / 2);
  atomicAdd(dev_ft_cluster_num, byte_count / 2);
  //printf("Event No: %u, Number of Raw Banks: %u, Version: %u, Total Rawbank Size: %u\n", event_id, event.number_of_raw_banks, event.version, byte_count);
}
