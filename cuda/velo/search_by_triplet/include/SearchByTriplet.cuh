#pragma once

#include <stdint.h>
#include "../../common/include/VeloDefinitions.cuh"
#include "FillCandidates.cuh"
#include "ProcessModules.cuh"
#include "TrackForwarding.cuh"
#include "TrackSeeding.cuh"
#include "TrackSeedingFirst.cuh"

__global__ void searchByTriplet(
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  TrackHits* dev_tracks,
  TrackHits* dev_tracklets,
  uint* dev_tracks_to_follow,
  uint* dev_weak_tracks,
  bool* dev_hit_used,
  int* dev_atomics_storage,
  short* dev_h0_candidates,
  short* dev_h2_candidates,
  unsigned short* dev_rel_indices
);
