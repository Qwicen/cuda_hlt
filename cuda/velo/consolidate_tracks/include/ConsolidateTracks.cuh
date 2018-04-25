#pragma once

#include "../../common/include/VeloDefinitions.cuh"

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const TrackHits* dev_tracks,
  TrackHits* dev_output_tracks
);
