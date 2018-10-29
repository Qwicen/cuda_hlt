#pragma once

#include "SciFiDefinitions.cuh"
#include "Handler.cuh"

__device__ void make_cluster (
  const int hit_index,
  const SciFi::SciFiHitCount& hit_count,
  const SciFi::SciFiGeometry& geom,
  uint32_t chan,
  uint8_t fraction,
  uint8_t pseudoSize,
  uint32_t uniqueMat,
  SciFi::SciFiHits& hits);

__global__ void scifi_raw_bank_decoder(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  uint *scifi_hits,
  char *scifi_geometry);

ALGORITHM(scifi_raw_bank_decoder, scifi_raw_bank_decoder_t)
