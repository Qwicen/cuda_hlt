/**
 * Specify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE_T(
  init_event_list_t,
  global_event_cut_t,

  // Velo data preparation
  velo_estimate_input_size_t,
  prefix_sum_velo_clusters_t,
  velo_masked_clustering_t,

  // UT data preparation
  ut_calculate_number_of_hits_t,
  prefix_sum_ut_hits_t,
  ut_pre_decode_t,
  ut_find_permutation_t,
  ut_decode_raw_banks_in_order_t,

  // SciFi data preparation
  scifi_calculate_cluster_count_v4_t,
  prefix_sum_scifi_hits_t,
  scifi_pre_decode_v4_t,
  scifi_raw_bank_decoder_v4_t,
  scifi_direct_decoder_v4_t)
