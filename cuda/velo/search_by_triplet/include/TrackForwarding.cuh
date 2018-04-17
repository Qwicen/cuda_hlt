#pragma once

#include "../../common/include/Definitions.cuh"

__device__ void trackForwarding(
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  bool* hit_used,
  unsigned int* tracks_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* weaktracks_insertPointer,
  const Module* module_data,
  const unsigned int diff_ttf,
  unsigned int* tracks_to_follow,
  unsigned int* weak_tracks,
  const unsigned int prev_ttf,
  Track* tracklets,
  Track* tracks,
  const unsigned int number_of_hits,
  const unsigned int first_module,
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums
);
