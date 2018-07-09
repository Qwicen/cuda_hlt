#pragma once

#include "../../common/include/VeloDefinitions.cuh"
#include "../../../../main/include/Common.h"
#include <stdint.h>

__device__ VeloState means_square_fit(
  const Hit<mc_check_enabled>* velo_track_hits,
  const TrackHits& track
);

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const TrackHits* dev_tracks,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states
);
