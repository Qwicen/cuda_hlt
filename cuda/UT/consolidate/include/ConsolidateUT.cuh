#pragma once

#include "UTDefinitions.cuh"
#include "UTEventModel.cuh"
#include "UTConsolidated.cuh"
#include "Handler.cuh"
#include "ArgumentsUT.cuh"

__global__ void consolidate_ut_tracks(
  uint* dev_ut_hits,
  uint* dev_ut_hit_offsets,
  char* dev_ut_track_hits,
  int* dev_atomics_ut,
  uint* dev_ut_track_hit_number,
  float* dev_ut_qop,
  uint* dev_ut_track_velo_indices,
  const UT::TrackHits* dev_veloUT_tracks,
  const uint* dev_unique_x_sector_layer_offsets);

ALGORITHM(
  consolidate_ut_tracks,
  consolidate_ut_tracks_t,
  ARGUMENTS(
    dev_ut_hits,
    dev_ut_hit_offsets,
    dev_ut_track_hits,
    dev_atomics_ut,
    dev_ut_track_hit_number,
    dev_ut_qop,
    dev_ut_track_velo_indices,
    dev_ut_tracks));
