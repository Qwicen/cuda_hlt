#include "SearchByTriplet.cuh"
#include "ClusteringDefinitions.cuh"

/**
 * @brief Track forwarding algorithm based on triplet finding
 */
__global__ void search_by_triplet(
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  Velo::TrackHits* dev_tracks,
  Velo::TrackletHits* dev_tracklets,
  uint* dev_tracks_to_follow,
  Velo::TrackletHits* dev_weak_tracks,
  bool* dev_hit_used,
  int* dev_atomics_storage,
  short* dev_h0_candidates,
  short* dev_h2_candidates,
  unsigned short* dev_rel_indices,
  const float* dev_velo_module_zs
) {
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const uint event_number = blockIdx.x;
  const uint number_of_events = gridDim.x;
  const uint tracks_offset = event_number * VeloTracking::max_tracks;

  // Pointers to data within the event
  const uint number_of_hits = dev_module_cluster_start[VeloTracking::n_modules * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * VeloTracking::n_modules;
  const uint* module_hitNums = dev_module_cluster_num + event_number * VeloTracking::n_modules;
  const uint hit_offset = module_hitStarts[0];
  assert((module_hitStarts[52] - module_hitStarts[0]) < VeloTracking::max_number_of_hits_per_event);
  
  // Order has changed since SortByPhi
  const float* hit_Ys = (float*) (dev_velo_cluster_container + hit_offset);
  const float* hit_Zs = (float*) (dev_velo_cluster_container + number_of_hits + hit_offset);
  const float* hit_Phis = (float*) (dev_velo_cluster_container + 4 * number_of_hits + hit_offset);
  const float* hit_Xs = (float*) (dev_velo_cluster_container + 5 * number_of_hits + hit_offset);
  // const float* hit_IDs = (float*) (dev_velo_cluster_container + 2 * number_of_hits + hit_offset);

  // Per event datatypes
  Velo::TrackHits* tracks = dev_tracks + tracks_offset;
  uint* tracks_insert_pointer = (uint*) dev_atomics_storage + event_number;

  // Per side datatypes
  bool* hit_used = dev_hit_used + hit_offset;
  short* h0_candidates = dev_h0_candidates + 2*hit_offset;
  short* h2_candidates = dev_h2_candidates + 2*hit_offset;

  uint* tracks_to_follow = dev_tracks_to_follow + event_number * VeloTracking::ttf_modulo;
  Velo::TrackletHits* weak_tracks = dev_weak_tracks + event_number * VeloTracking::max_weak_tracks;
  Velo::TrackletHits* tracklets = dev_tracklets + event_number * VeloTracking::ttf_modulo;
  unsigned short* h1_rel_indices = dev_rel_indices + event_number * VeloTracking::max_numhits_in_module;

  // Initialize variables according to event number and module side
  // Insert pointers (atomics)
  const int ip_shift = number_of_events + event_number * (VeloTracking::num_atomics - 1);
  uint* weaktracks_insert_pointer = (uint*) dev_atomics_storage + ip_shift;
  uint* tracklets_insert_pointer = (uint*) dev_atomics_storage + ip_shift + 1;
  uint* ttf_insert_pointer = (uint*) dev_atomics_storage + ip_shift + 2;
  uint* local_number_of_hits = (uint*) dev_atomics_storage + ip_shift + 3;

  // Shared memory
  extern __shared__ float shared_best_fits [];
  __shared__ int module_data [18];

  // Initialize hit_used
  const auto current_event_number_of_hits = module_hitStarts[VeloTracking::n_modules] - hit_offset;
  for (int i=0; i<(current_event_number_of_hits + blockDim.x - 1) / blockDim.x; ++i) {
    const auto index = i*blockDim.x + threadIdx.x;
    if (index < current_event_number_of_hits) {
      hit_used[index] = false;
    }
  }
  // Initialize atomics
  tracks_insert_pointer[0] = 0;
  if (threadIdx.x < (VeloTracking::num_atomics - 1)) {
    dev_atomics_storage[ip_shift + threadIdx.x] = 0;
  }

  // Process modules
  process_modules(
    (Velo::Module*) &module_data[0],
    (float*) &shared_best_fits[0],
    VP::NModules-1,
    2,
    hit_used,
    h0_candidates,
    h2_candidates,
    VP::NModules,
    module_hitStarts,
    module_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Zs,
    hit_Phis,
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
    hit_offset,
    dev_velo_module_zs
  );
}
