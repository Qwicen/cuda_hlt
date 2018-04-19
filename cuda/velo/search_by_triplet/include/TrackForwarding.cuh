#pragma once

#include "VeloDefinitions.cuh"

__device__ void trackForwarding(
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  bool* hit_used,
  uint* tracks_insertPointer,
  uint* ttf_insertPointer,
  uint* weaktracks_insertPointer,
  const Module* module_data,
  const uint diff_ttf,
  uint* tracks_to_follow,
  uint* weak_tracks,
  const uint prev_ttf,
  Track* tracklets,
  Track* tracks,
  const uint number_of_hits
);
