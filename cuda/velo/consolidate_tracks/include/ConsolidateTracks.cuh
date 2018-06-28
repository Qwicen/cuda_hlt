#pragma once

#include "../../common/include/VeloDefinitions.cuh"
#include "../../../../main/include/Common.h"
#include <stdint.h>

__device__ void means_square_fit(
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const TrackHits& track,
  VeloState& state
);

__device__ Track<mc_check_enabled> createTrack(
  const TrackHits &track,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const uint32_t* hit_IDs
);

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const TrackHits* dev_tracks,
  Track<mc_check_enabled>* dev_output_tracks,
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  VeloState* dev_velo_states
);
