/**
 * Specify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE_T(
  // Velo data preparation
  estimate_input_size_t,
  prefix_sum_reduce_velo_clusters_t,
  prefix_sum_single_block_velo_clusters_t,
  prefix_sum_scan_velo_clusters_t,
  masked_velo_clustering_t,
  
  // UT data preparation
  ut_calculate_number_of_hits_t,
  prefix_sum_reduce_ut_hits_t,
  prefix_sum_single_block_ut_hits_t,
  prefix_sum_scan_ut_hits_t,
  ut_pre_decode_t,
  ut_find_permutation_t,
  ut_decode_raw_banks_in_order_t,
  
  // SciFi data preparation
  estimate_cluster_count_t,
  prefix_sum_reduce_scifi_hits_t,
  prefix_sum_single_block_scifi_hits_t,
  prefix_sum_scan_scifi_hits_t,
  raw_bank_decoder_t,
  scifi_sort_by_x_t
)
