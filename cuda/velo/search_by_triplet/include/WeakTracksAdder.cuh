#pragma once

#include "../../common/include/VeloDefinitions.cuh"

__device__ void weak_tracks_adder_impl(
  uint* weaktracks_insert_pointer,
  uint* tracks_insert_pointer,
  TrackHits* weak_tracks,
  TrackHits* tracks,
  bool* hit_used,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs
);

__global__ void weak_tracks_adder(
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  TrackHits* dev_tracks,
  TrackHits* dev_weak_tracks,
  bool* dev_hit_used,
  int* dev_atomics_storage
);
