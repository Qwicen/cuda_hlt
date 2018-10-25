#pragma once

#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"
#include "Common.h"
#include "Handler.cuh"
#include <cstdint>

__device__ Velo::State means_square_fit(
  Velo::Consolidated::Hits& consolidated_hits,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const uint* hit_IDs,
  const Velo::TrackHits& track
);

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const Velo::TrackHits* dev_tracks,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint* dev_velo_track_hits,
  uint* dev_velo_states
);

ALGORITHM(consolidate_tracks, consolidate_tracks_t)
