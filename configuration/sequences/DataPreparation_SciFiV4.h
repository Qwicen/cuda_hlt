/**
 * Specify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE_T(
  // SciFi data preparation
  scifi_calculate_cluster_count_v4_t,
  prefix_sum_scifi_hits_t,
  scifi_pre_decode_v4_t,
  scifi_raw_bank_decoder_v4_t,
  scifi_direct_decoder_v4_t
)
