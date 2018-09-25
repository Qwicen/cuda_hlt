#pragma once

#include "VeloEventModel.cuh"
#include <cassert>

__device__ void process_modules(
  Velo:: Module* module_data,
  float* shared_best_fits,
  const uint starting_module,
  const uint stride,
  bool* hit_used,
  const short* h0_candidates,
  const short* h2_candidates,
  const uint number_of_modules,
  const uint* module_hitStarts,
  const uint* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const float* hit_Phis,
  uint* weaktracks_insert_pointer,
  uint* tracklets_insert_pointer,
  uint* ttf_insert_pointer,
  uint* tracks_insert_pointer,
  uint* tracks_to_follow,
  Velo::TrackletHits* weak_tracks,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  const uint number_of_hits,
  unsigned short* h1_rel_indices,
  uint* local_number_of_hits,
  const uint hit_offset,
  const float* dev_velo_module_zs
);
