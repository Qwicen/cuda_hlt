#pragma once

#include "VeloEventModel.cuh"

__device__ void track_forwarding(
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  bool* hit_used,
  uint* tracks_insertPointer,
  uint* ttf_insertPointer,
  uint* weaktracks_insertPointer,
  const Velo::Module* module_data,
  const uint diff_ttf,
  uint* tracks_to_follow,
  Velo::TrackletHits* weak_tracks,
  const uint prev_ttf,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  const uint number_of_hits
);
