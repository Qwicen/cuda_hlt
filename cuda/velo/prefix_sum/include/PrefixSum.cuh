#pragma once

#include "../../common/include/VeloDefinitions.cuh"

__global__ void prefix_sum_reduce(
  uint* dev_estimated_input_size,
  uint* dev_cluster_offset,
  const uint array_size
);

__global__ void prefix_sum_single_block(
  uint* dev_total_sum,
  uint* dev_array,
  const uint array_size
);

__global__ void copy_and_prefix_sum_single_block(
  uint* dev_total_sum,
  uint* dev_input_array,
  uint* dev_output_array,
  const uint array_size
);

__global__ void copy_and_ps_velo_track_hit_number(
  const TrackHits* dev_tracks,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  const uint number_of_events
);

__global__ void prefix_sum_scan(
  uint* dev_estimated_input_size,
  uint* dev_cluster_offset,
  const uint array_size
);
