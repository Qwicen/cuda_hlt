#pragma once

#include "VeloDefinitions.cuh"

__device__ void weak_tracks_adder_impl(
  uint* weaktracks_insert_pointer,
  uint* tracks_insert_pointer,
  VeloTracking::TrackletHits* weak_tracks,
  VeloTracking::TrackHits* tracks,
  bool* hit_used,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs
);

__global__ void weak_tracks_adder(
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  VeloTracking::TrackHits* dev_tracks,
  VeloTracking::TrackletHits* dev_weak_tracks,
  bool* dev_hit_used,
  int* dev_atomics_storage
);
