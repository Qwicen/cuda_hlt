#pragma once

#include "../../common/include/VeloDefinitions.cuh"

__device__ void track_seeding(
  float* shared_best_fits,
  const float* hit_Xs,
  const float* hit_Ys,
  const Module* module_data,
  const short* h0_candidates,
  const short* h2_candidates,
  bool* hit_used,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  TrackHits* tracklets,
  unsigned int* tracks_to_follow,
  unsigned short* h1_rel_indices,
  unsigned int* local_number_of_hits
);
