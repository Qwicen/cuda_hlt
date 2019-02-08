#pragma once

#include "SciFiEventModel.cuh"
#include "SciFiConsolidated.cuh"
#include "SciFiDefinitions.cuh"
#include "MiniState.cuh"
#include "Handler.cuh"
#include "ArgumentsSciFi.cuh"

__global__ void consolidate_scifi_tracks(
  uint* dev_scifi_hits,
  uint* dev_scifi_hit_count,
  char* dev_scifi_track_hits,
  int* dev_atomics_scifi,
  uint* dev_scifi_track_hit_number,
  float* dev_scifi_qop,
  MiniState* dev_scifi_states,
  uint* dev_ut_indices,
  SciFi::TrackHits* dev_scifi_tracks,
  const char* dev_scifi_geometry,
  const float* dev_inv_clus_res);

ALGORITHM(
  consolidate_scifi_tracks,
  consolidate_scifi_tracks_t,
  ARGUMENTS(
    dev_scifi_hits,
    dev_scifi_hit_count,
    dev_scifi_track_hits,
    dev_atomics_scifi,
    dev_scifi_track_hit_number,
    dev_scifi_qop,
    dev_scifi_states,
    dev_scifi_track_ut_indices,
    dev_scifi_tracks))
