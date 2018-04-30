#pragma once

#include "../../common/include/VeloDefinitions.cuh"
#include <stdint.h>

template <bool mc_check>
__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const TrackHits* dev_tracks,
  Track <mc_check> * dev_output_tracks,
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num
);

template <bool mc_check>
__device__ Track <mc_check> createTrack(
  const TrackHits &track,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_zs,
  const uint32_t* hit_IDs
);
