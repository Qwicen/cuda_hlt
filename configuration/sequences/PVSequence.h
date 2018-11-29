/**
 * Specify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE_T(
  velo_estimate_input_size_t,
  prefix_sum_reduce_velo_clusters_t,
  prefix_sum_single_block_velo_clusters_t,
  prefix_sum_scan_velo_clusters_t,
  velo_masked_clustering_t,
  velo_calculate_phi_and_sort_t,
  velo_fill_candidates_t,
  velo_search_by_triplet_t,
  velo_weak_tracks_adder_t,
  copy_and_prefix_sum_single_block_velo_t,
  copy_velo_track_hit_number_t,
  prefix_sum_reduce_velo_track_hit_number_t,
  prefix_sum_single_block_velo_track_hit_number_t,
  prefix_sum_scan_velo_track_hit_number_t,
  consolidate_velo_tracks_t,
  velo_kalman_fit_t,
  pv_get_seeds_t,
  pv_fit_seeds_t
)
