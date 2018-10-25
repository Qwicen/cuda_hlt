#pragma once

namespace arg {
/**
 * arg_enum_t Arguments for all algorithms in the sequence.
 */
enum arg_enum_t {
  dev_raw_input,
  dev_raw_input_offsets,
  dev_estimated_input_size,
  dev_module_cluster_num,
  dev_module_candidate_num,
  dev_cluster_offset,
  dev_cluster_candidates,
  dev_velo_cluster_container,
  dev_tracks,
  dev_tracks_to_follow,
  dev_hit_used,
  dev_atomics_storage,
  dev_tracklets,
  dev_weak_tracks,
  dev_h0_candidates,
  dev_h2_candidates,
  dev_rel_indices,
  dev_hit_permutation,
  dev_velo_track_hit_number,
  dev_prefix_sum_auxiliary_array_2,
  dev_velo_track_hits,
  dev_velo_states,
  dev_ut_raw_input,
  dev_ut_raw_input_offsets,
  dev_ut_hit_offsets,
  dev_ut_hit_count,
  dev_prefix_sum_auxiliary_array_3,
  dev_ut_hits,
  dev_ut_hit_permutations,
  dev_veloUT_tracks,
  dev_atomics_veloUT,
  dev_scifi_raw_input_offsets,
  dev_scifi_hit_count,
  dev_prefix_sum_auxiliary_array_4,
  dev_scifi_hit_permutations,
  dev_scifi_hits,
  dev_scifi_raw_input,
  dev_scifi_tracks,
  dev_n_scifi_tracks
};
}
