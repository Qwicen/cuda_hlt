#pragma once

#include "../../common/include/VeloDefinitions.cuh"

__device__ void weak_tracks_adder(
  uint* weaktracks_insert_pointer,
  uint* tracks_insert_pointer,
  VeloTracking::TrackHits* weak_tracks,
  VeloTracking::TrackHits* tracks,
  bool* hit_used,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs
);
