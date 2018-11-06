/**
 * Specify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE_T(
  estimate_input_size_t,
  prefix_sum_reduce_velo_clusters_t,
  prefix_sum_single_block_velo_clusters_t,
  prefix_sum_scan_velo_clusters_t,
  masked_velo_clustering_t,
  calculate_phi_and_sort_t,
  fill_candidates_t,
  search_by_triplet_t,
  weak_tracks_adder_t,
  copy_and_prefix_sum_single_block_t,
  copy_velo_track_hit_number_t,
  prefix_sum_reduce_velo_track_hit_number_t,
  prefix_sum_single_block_velo_track_hit_number_t,
  prefix_sum_scan_velo_track_hit_number_t,
  consolidate_tracks_t,
  ut_calculate_number_of_hits_t,
  prefix_sum_reduce_ut_hits_t,
  prefix_sum_single_block_ut_hits_t,
  prefix_sum_scan_ut_hits_t,
  ut_pre_decode_t,
  ut_find_permutation_t,
  ut_decode_raw_banks_in_order_t
)
