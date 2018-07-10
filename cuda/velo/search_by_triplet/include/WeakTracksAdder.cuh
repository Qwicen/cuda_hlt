#pragma once

#include "../../common/include/VeloDefinitions.cuh"

__device__ void weakTracksAdder(
  uint* weaktracks_insert_pointer,
  uint* tracks_insert_pointer,
  TrackHits* weak_tracks,
  TrackHits* tracks,
  bool* hit_used,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs
);
