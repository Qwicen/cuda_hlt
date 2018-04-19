#pragma once

#include "VeloDefinitions.cuh"

__device__ void trackSeedingFirst(
  float* shared_best_fits,
  const float* hit_Xs,
  const float* hit_Ys,
  const Module* module_data,
  const short* h0_candidates,
  const short* h2_candidates,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  Track* tracklets,
  unsigned int* tracks_to_follow
);
