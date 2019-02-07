#pragma once

#include "VeloEventModel.cuh"
#include "Handler.cuh"
#include "ArgumentsVelo.cuh"
#include "States.cuh"

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
  int* dev_atomics_velo
);

ALGORITHM(weak_tracks_adder, velo_weak_tracks_adder_t,
  ARGUMENTS(
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_tracks,
    dev_weak_tracks,
    dev_hit_used,
    dev_atomics_velo
))
