#pragma once

#include "../../common/include/Definitions.cuh"

__device__ void processModules(
  Module* module_data,
  float* shared_best_fits,
  const unsigned int starting_module,
  const unsigned int stride,
  bool* hit_used,
  const short* h0_candidates,
  const short* h2_candidates,
  const unsigned int number_of_modules,
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  unsigned int* weaktracks_insertPointer,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  unsigned int* tracks_insertPointer,
  unsigned int* tracks_to_follow,
  unsigned int* weak_tracks,
  Track* tracklets,
  Track* tracks,
  const unsigned int number_of_hits,
  unsigned short* h1_rel_indices,
  unsigned int* local_number_of_hits
);
