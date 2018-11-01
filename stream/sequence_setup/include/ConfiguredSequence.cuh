#pragma once

#include <tuple>
#include "CalculatePhiAndSort.cuh"
#include "ConsolidateTracks.cuh"
#include "MaskedVeloClustering.cuh"
#include "EstimateInputSize.cuh"
#include "PrefixSum.cuh"
#include "SearchByTriplet.cuh"
#include "VeloKalmanFilter.cuh"
#include "VeloUT.cuh"
#include "EstimateClusterCount.cuh"
#include "RawBankDecoder.cuh"
#include "SciFiSortByX.cuh"
#include "VeloEventModel.cuh"
#include "UTCalculateNumberOfHits.cuh"
#include "UTDecodeRawBanksInOrder.cuh"
#include "UTFindPermutation.cuh"
#include "UTPreDecode.cuh"
#include "PrForward.cuh"

#define SEQUENCE(...) \
  typedef std::tuple<__VA_ARGS__> sequence_t;
  // Prepared for C++17 variant
  // typedef std::variant<std::monostate, __VA_ARGS__> state_t;

/**
 * Especify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE(
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
  ut_decode_raw_banks_in_order_t,
  veloUT_t,
  estimate_cluster_count_t,
  prefix_sum_reduce_scifi_hits_t,
  prefix_sum_single_block_scifi_hits_t,
  prefix_sum_scan_scifi_hits_t,
  raw_bank_decoder_t,
  scifi_sort_by_x_t,
  scifi_pr_forward_t
)
