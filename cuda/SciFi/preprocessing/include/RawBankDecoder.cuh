#pragma once

#include "SciFiDefinitions.cuh"
#include "Handler.cuh"

__global__ void raw_bank_decoder(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  uint *scifi_hits,
  char *scifi_geometry);

__device__ uint32_t channelInBank(uint32_t c);
__device__ uint16_t getLinkInBank(uint16_t c);
__device__ int cell(uint16_t c);
__device__ int fraction(uint16_t c);
__device__ bool cSize(uint16_t c);

ALGORITHM(raw_bank_decoder, raw_bank_decoder_t)
