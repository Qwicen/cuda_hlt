#pragma once

#include "SciFiDefinitions.cuh"
#include "Handler.cuh"

__device__ void store_sorted_cluster_reference (
  const SciFi::SciFiHitCount& hit_count,
  const uint32_t uniqueMat,
  const uint32_t chan,
  const uint32_t* shared_mat_offsets,
  const int raw_bank,
  const int it,
  const int condition_1,
  const int condition_2,
  const int delta,
  SciFi::SciFiHits& hits);

__global__ void scifi_pre_decode(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  char *scifi_hits,
  char *scifi_geometry);

ALGORITHM(scifi_pre_decode, scifi_pre_decode_t)
