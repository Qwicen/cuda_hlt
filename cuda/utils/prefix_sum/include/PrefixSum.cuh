#pragma once

#include "VeloEventModel.cuh"
#include "Handler.cuh"

__global__ void prefix_sum_reduce(
  uint* dev_main_array,
  uint* dev_auxiliary_array,
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

__global__ void prefix_sum_scan(
  uint* dev_main_array,
  uint* dev_auxiliary_array,
  const uint array_size
);

__global__ void copy_velo_track_hit_number(
  const Velo::TrackHits* dev_tracks,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number
);

ALGORITHM(prefix_sum_reduce, prefix_sum_reduce_velo_clusters_t)
ALGORITHM(prefix_sum_single_block, prefix_sum_single_block_velo_clusters_t)
ALGORITHM(prefix_sum_scan, prefix_sum_scan_velo_clusters_t)

ALGORITHM(prefix_sum_reduce, prefix_sum_reduce_velo_track_hit_number_t)
ALGORITHM(prefix_sum_single_block, prefix_sum_single_block_velo_track_hit_number_t)
ALGORITHM(prefix_sum_scan, prefix_sum_scan_velo_track_hit_number_t)

ALGORITHM(prefix_sum_reduce, prefix_sum_reduce_ut_hits_t)
ALGORITHM(prefix_sum_single_block, prefix_sum_single_block_ut_hits_t)
ALGORITHM(prefix_sum_scan, prefix_sum_scan_ut_hits_t)

ALGORITHM(prefix_sum_reduce, prefix_sum_reduce_scifi_hits_t)
ALGORITHM(prefix_sum_single_block, prefix_sum_single_block_scifi_hits_t)
ALGORITHM(prefix_sum_scan, prefix_sum_scan_scifi_hits_t)

ALGORITHM(copy_and_prefix_sum_single_block, copy_and_prefix_sum_single_block_t)
ALGORITHM(copy_velo_track_hit_number, copy_velo_track_hit_number_t)
