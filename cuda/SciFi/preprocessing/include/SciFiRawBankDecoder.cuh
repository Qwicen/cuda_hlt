#pragma once

#include "SciFiDefinitions.cuh"
#include "Handler.cuh"

__global__ void scifi_raw_bank_decoder(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  char *scifi_hits,
  char *scifi_geometry);

ALGORITHM(scifi_raw_bank_decoder, scifi_raw_bank_decoder_t)
