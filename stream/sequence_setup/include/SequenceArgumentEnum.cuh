#pragma once

namespace seq {
/**
 * seq_enum_t contains all steps of the sequence in the expected
 *            order of execution.
 */
enum seq_enum_t {
  estimate_input_size,
  prefix_sum_reduce,
  prefix_sum_single_block,
  prefix_sum_scan,
  masked_velo_clustering,
  calculate_phi_and_sort,
  fill_candidates,
  search_by_triplet,
  weak_tracks_adder,
  copy_and_prefix_sum_single_block,
  copy_velo_track_hit_number,
  prefix_sum_reduce_velo_track_hit_number,
  prefix_sum_single_block_velo_track_hit_number,
  prefix_sum_scan_velo_track_hit_number,
  consolidate_tracks,
  sort_by_x,
  veloUT
};
}

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
  dev_ut_hits,
  dev_ut_hits_sorted,
  dev_ut_hit_permutations,
  dev_veloUT_tracks,
  dev_atomics_veloUT
};
}
