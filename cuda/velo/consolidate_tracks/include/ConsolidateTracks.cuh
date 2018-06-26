#pragma once

#include "../../common/include/VeloDefinitions.cuh"
#include <stdint.h>

__device__ void means_square_fit(
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const VeloTracking::TrackHits& track,
  VeloState& state
);

template<bool mc_check_enabled>
__device__ VeloTracking::Track<mc_check_enabled> createTrack(
  const VeloTracking::TrackHits &track,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const uint32_t* hit_IDs
);

template<bool mc_check_enabled>
__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const VeloTracking::TrackHits* dev_tracks,
  VeloTracking::Track<mc_check_enabled>* dev_output_tracks,
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  VeloState* dev_velo_states
);

#include "ConsolidateTracks_impl.cuh"
