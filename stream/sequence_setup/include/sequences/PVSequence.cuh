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
  velo_kalman_fit_t,
  //getSeeds_t,
  //fitSeeds_t
  cpu_beamlinePV_t,
  blpv_extrapolate_t,
  blpv_histo_t
)
