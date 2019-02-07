#pragma once

#include <cstdint>
#include <cfloat>
#include "VeloEventModel.cuh"
#include "FillCandidates.cuh"
#include "ProcessModules.cuh"
#include "TrackForwarding.cuh"
#include "TrackSeeding.cuh"
#include "WeakTracksAdder.cuh"
#include "Handler.cuh"
#include "ArgumentsVelo.cuh"

__global__ void search_by_triplet(
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  Velo::TrackHits* dev_tracks,
  Velo::TrackletHits* dev_tracklets,
  uint* dev_tracks_to_follow,
  Velo::TrackletHits* dev_weak_tracks,
  bool* dev_hit_used,
  int* dev_atomics_velo,
  short* dev_h0_candidates,
  short* dev_h2_candidates,
  unsigned short* dev_rel_indices,
  const float* dev_velo_module_zs
);

ALGORITHM(search_by_triplet, velo_search_by_triplet_t,
  ARGUMENTS(
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_tracks,
    dev_tracklets,
    dev_tracks_to_follow,
    dev_weak_tracks,
    dev_hit_used,
    dev_atomics_velo,
    dev_h0_candidates,
    dev_h2_candidates,
    dev_rel_indices
))
