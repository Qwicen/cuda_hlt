#include "../include/SearchByTriplet.cuh"

void SearchByTriplet::operator()() {
  searchByTriplet<<<num_blocks, num_threads, 0, *stream>>>(
    dev_velo_cluster_container,
    dev_module_cluster_start,
    dev_module_cluster_num,
    dev_tracks,
    dev_tracklets,
    dev_tracks_to_follow,
    dev_weak_tracks,
    dev_hit_used,
    dev_atomics_storage,
    dev_h0_candidates,
    dev_h2_candidates,
    dev_rel_indices
  );
}
