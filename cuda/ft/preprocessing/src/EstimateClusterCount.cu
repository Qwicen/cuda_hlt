#include "EstimateClusterCount.cuh"
#include <stdio.h>
#include "assert.h"

__global__ void estimate_cluster_count(uint *ft_event_offsets, uint *dev_ft_cluster_count, char *ft_events) {
  //printf("Preprocessing FT event %u, offset: %u...\n", blockIdx.x, ft_event_offsets[blockIdx.x]);
  //printf("blockIdx.x = %u\n", blockIdx.x);
  const uint event_id = blockIdx.x;
  const auto event = FTRawEvent(ft_events + ft_event_offsets[event_id]);
  printf("Event No: %u, Number of Raw Banks: %u, Version: %u\n", event_id, event.number_of_raw_banks, event.version);
  assert(event.version == 5u);
  for(size_t rawbank = 0; rawbank < event.number_of_raw_banks; rawbank++)
  {
    atomicAdd(dev_ft_cluster_count + event_id, event.raw_bank_offset[rawbank]);
  }
  //dev_ft_cluster_count[event_id] = event.number_of_raw_banks
}

__device__ __host__ FTRawEvent::FTRawEvent(
  const char* event
) {
  const char* p = event;
  number_of_raw_banks = *((uint32_t*)p); p += sizeof(uint32_t);
  version = *((uint32_t*)p); p += sizeof(uint32_t);
  raw_bank_offset = (uint32_t*) p; p += number_of_raw_banks * sizeof(uint32_t);
  payload = (char*) p;
}

__device__ __host__ FTRawBank::FTRawBank(
  const char* raw_bank,
  unsigned int len
) {
  const char* p = raw_bank;
  sourceID = *((uint32_t*)p); p += sizeof(uint32_t);
  length = len;
  data = (uint32_t*) p;
}
