#pragma once

#include "../../common/include/VeloDefinitions.cuh"
#include "../../common/include/ClusteringDefinitions.cuh"

__device__ void track_seeding_first(
  float* shared_best_fits,
  const float* hit_Xs,
  const float* hit_Ys,
  const VeloTracking::Module* module_data,
  const short* h0_candidates,
  const short* h2_candidates,
  unsigned int* tracklets_insertPointer,
  unsigned int* ttf_insertPointer,
  VeloTracking::TrackHits* tracklets,
  unsigned int* tracks_to_follow
);
