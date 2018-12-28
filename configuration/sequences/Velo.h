/**
 * Specify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE_T(
  init_event_list_t,
  global_event_cut_t,
  velo_estimate_input_size_t,
  prefix_sum_velo_clusters_t,
  velo_masked_clustering_t,
  velo_calculate_phi_and_sort_t,
  velo_fill_candidates_t,
  velo_search_by_triplet_t,
  velo_weak_tracks_adder_t,
  copy_and_prefix_sum_single_block_velo_t,
  copy_velo_track_hit_number_t,
  prefix_sum_velo_track_hit_number_t,
  consolidate_velo_tracks_t
)
