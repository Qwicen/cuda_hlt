#pragma once

#include "SciFiEventModel.cuh"
#include "SciFiConsolidated.cuh"
#include "SciFiDefinitions.cuh"

#include "Handler.cuh"

__global__ void consolidate_scifi_tracks(
  uint* dev_scifi_hits,
  uint* dev_scifi_hit_count,
  char* dev_scifi_track_hits,
  int* dev_atomics_scifi,
  uint* dev_scifi_track_hit_number,
  float* dev_scifi_qop,
  uint* dev_ut_indices,
  SciFi::TrackHits* dev_scifi_tracks,
  const char* dev_scifi_geometry, 
  const float* dev_inv_clus_res
);

ALGORITHM(consolidate_scifi_tracks, consolidate_scifi_tracks_t)

