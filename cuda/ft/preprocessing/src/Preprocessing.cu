#include "Preprocessing.cuh"
#include <stdio.h>

__global__ void preprocessing(uint *ft_event_offsets, char *ft_events) {
  printf("Preprocessing FT event %u, offset: %u...\n", blockIdx.x, ft_event_offsets[blockIdx.x]);
  const uint event_id = blockIdx.x;
  const auto event = FTRawEvent(ft_events + ft_event_offsets[event_id]);
  printf("Number of Raw Banks: %u, Version: %u\n", event.number_of_raw_banks, event.version);
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
