#pragma once

#include "VeloEventModel.cuh"

__device__ void track_seeding(
  float* shared_best_fits,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const Velo::Module* module_data,
  const short* h0_candidates,
  const short* h2_candidates,
  bool* hit_used,
  uint* tracklets_insertPointer,
  uint* ttf_insertPointer,
  Velo::TrackletHits* tracklets,
  uint* tracks_to_follow,
  unsigned short* h1_rel_indices,
  uint* local_number_of_hits);
