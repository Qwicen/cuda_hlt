#pragma once

#include "VeloEventModel.cuh"
#include "Handler.cuh"

__device__ void weak_tracks_adder_impl(
  uint* weaktracks_insert_pointer,
  uint* tracks_insert_pointer,
  Velo::TrackletHits* weak_tracks,
  Velo::TrackHits* tracks,
  bool* hit_used,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs
);

__global__ void weak_tracks_adder(
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  Velo::TrackHits* dev_tracks,
  Velo::TrackletHits* dev_weak_tracks,
  bool* dev_hit_used,
  int* dev_atomics_storage
);

ALGORITHM(weak_tracks_adder, weak_tracks_adder_t)
