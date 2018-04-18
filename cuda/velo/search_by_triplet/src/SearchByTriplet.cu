﻿#include "../include/SearchByTriplet.cuh"
#include "../../common/include/ClusteringDefinitions.cuh"

/**
 * @brief Track forwarding algorithm based on triplet finding
 */
__global__ void searchByTriplet(
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  Track* dev_tracks,
  Track* dev_tracklets,
  uint* dev_tracks_to_follow,
  uint* dev_weak_tracks,
  bool* dev_hit_used,
  int* dev_atomics_storage,
  short* dev_h0_candidates,
  short* dev_h2_candidates,
  unsigned short* dev_rel_indices
) {
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const uint event_number = blockIdx.x;
  const uint number_of_events = gridDim.x;
  const uint tracks_offset = event_number * MAX_TRACKS;

  // Pointers to data within the event
  const uint number_of_hits = dev_module_cluster_start[52 * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * 52;
  const uint* module_hitNums = dev_module_cluster_num + event_number * 52;
  const uint hit_offset = module_hitStarts[0];
  
  // Order has changed since SortByPhi
  const float* hit_Ys = (float*) (dev_velo_cluster_container + hit_offset);
  const float* hit_Zs = (float*) (dev_velo_cluster_container + number_of_hits + hit_offset);
  const float* hit_Phis = (float*) (dev_velo_cluster_container + 4 * number_of_hits + hit_offset);
  const float* hit_Xs = (float*) (dev_velo_cluster_container + 5 * number_of_hits + hit_offset);

  // Per event datatypes
  Track* tracks = dev_tracks + tracks_offset;
  uint* tracks_insert_pointer = (uint*) dev_atomics_storage + event_number;

  // Per side datatypes
  bool* hit_used = dev_hit_used + hit_offset;
  short* h0_candidates = dev_h0_candidates + 2*hit_offset;
  short* h2_candidates = dev_h2_candidates + 2*hit_offset;

  uint* tracks_to_follow = dev_tracks_to_follow + event_number * TTF_MODULO;
  uint* weak_tracks = dev_weak_tracks + hit_offset;
  Track* tracklets = dev_tracklets + hit_offset;
  unsigned short* h1_rel_indices = dev_rel_indices + event_number * MAX_NUMHITS_IN_MODULE;

  // Initialize variables according to event number and module side
  // Insert pointers (atomics)
  const int ip_shift = number_of_events + event_number * NUM_ATOMICS;
  uint* weaktracks_insert_pointer = (uint*) dev_atomics_storage + ip_shift;
  uint* tracklets_insert_pointer = (uint*) dev_atomics_storage + ip_shift + 1;
  uint* ttf_insert_pointer = (uint*) dev_atomics_storage + ip_shift + 2;
  uint* local_number_of_hits = (uint*) dev_atomics_storage + ip_shift + 3;

  // Shared memory
  __shared__ float shared_best_fits [NUMTHREADS_X];
  __shared__ int module_data [9];

  // Initialize hit_used
  const auto current_event_number_of_hits = module_hitStarts[52] - hit_offset;
  for (int i=0; i<(current_event_number_of_hits + blockDim.x - 1) / blockDim.x; ++i) {
    const auto index = i*blockDim.x + threadIdx.x;
    if (index < current_event_number_of_hits) {
      hit_used[index] = false;
    }
  }
  // Initialize atomics
  tracks_insert_pointer[0] = 0;
  if (threadIdx.x < NUM_ATOMICS) {
    dev_atomics_storage[ip_shift + threadIdx.x] = 0;
  }

  // Fill candidates for both sides
  fillCandidates(
    h0_candidates,
    h2_candidates,
    module_hitStarts,
    module_hitNums,
    hit_Phis,
    hit_offset
  );

  // Process modules
  processModules(
    (Module*) &module_data[0],
    (float*) &shared_best_fits[0],
    VP::NModules-1,
    1,
    hit_used,
    h0_candidates,
    h2_candidates,
    VP::NModules,
    module_hitStarts,
    module_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Zs,
    weaktracks_insert_pointer,
    tracklets_insert_pointer,
    ttf_insert_pointer,
    tracks_insert_pointer,
    tracks_to_follow,
    weak_tracks,
    tracklets,
    tracks,
    number_of_hits,
    h1_rel_indices,
    local_number_of_hits,
    hit_offset
  );

  __syncthreads();

  // Process left weak tracks
  weakTracksAdder(
    (int*) &shared_best_fits[0],
    weaktracks_insert_pointer,
    tracks_insert_pointer,
    weak_tracks,
    tracklets,
    tracks,
    hit_used
  );
}