#include "../include/SearchByTriplet.cuh"

void SearchByTriplet::operator()() {
  searchByTriplet<<<num_blocks, num_threads, 0, stream>>>(
    dev_tracks,
    dev_events,
    dev_tracks_to_follow,
    dev_hit_used,
    dev_atomics_storage,
    dev_tracklets,
    dev_weak_tracks,
    dev_event_offsets,
    dev_hit_offsets,
    dev_h0_candidates,
    dev_h2_candidates,
    dev_rel_indices,
    dev_hit_phi,
    dev_hit_temp
  );
}
