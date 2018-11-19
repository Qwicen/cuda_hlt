#pragma once

#include <tuple>
#include "ConfiguredSequence.cuh"
#include "Arguments.cuh"

/**
 * @brief Definition of the dependencies of each algorithm.
 * @details All the dependencies for all defined algorithms
 *          should be defined here, using the type
 *          AlgorithmDependencies<Algorithm, Arguments...>.
 *          The types referred here are GPU buffers.
 */
typedef std::tuple<
  AlgorithmDependencies<estimate_input_size_t, // Algorithm
    dev_raw_input,                             // Argument #0
    dev_raw_input_offsets,                     // Argument #1
    dev_estimated_input_size,                  // ...
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates
  >,
  AlgorithmDependencies<prefix_sum_reduce_velo_clusters_t,
    dev_estimated_input_size,
    dev_cluster_offset
  >,
  AlgorithmDependencies<prefix_sum_single_block_velo_clusters_t,
    dev_estimated_input_size,
    dev_cluster_offset
  >,
  AlgorithmDependencies<prefix_sum_scan_velo_clusters_t,
    dev_estimated_input_size,
    dev_cluster_offset
  >,
  AlgorithmDependencies<masked_velo_clustering_t,
    dev_raw_input,
    dev_raw_input_offsets,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates,
    dev_velo_cluster_container
  >,
  AlgorithmDependencies<calculate_phi_and_sort_t,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_velo_cluster_container,
    dev_hit_permutation
  >,
  AlgorithmDependencies<fill_candidates_t,
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_h0_candidates,
    dev_h2_candidates
  >,
  AlgorithmDependencies<search_by_triplet_t,
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_tracks,
    dev_tracklets,
    dev_tracks_to_follow,
    dev_weak_tracks,
    dev_hit_used,
    dev_atomics_storage,
    dev_h0_candidates,
    dev_h2_candidates,
    dev_rel_indices
  >,
  AlgorithmDependencies<weak_tracks_adder_t,
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_tracks,
    dev_weak_tracks,
    dev_hit_used,
    dev_atomics_storage
  >,
  AlgorithmDependencies<copy_and_prefix_sum_single_block_t,
    dev_atomics_storage
  >,
  AlgorithmDependencies<copy_velo_track_hit_number_t,
    dev_tracks,
    dev_atomics_storage,
    dev_velo_track_hit_number
  >,
  AlgorithmDependencies<prefix_sum_reduce_velo_track_hit_number_t,
    dev_velo_track_hit_number,
    dev_prefix_sum_auxiliary_array_2
  >,
  AlgorithmDependencies<prefix_sum_single_block_velo_track_hit_number_t,
    dev_velo_track_hit_number,
    dev_prefix_sum_auxiliary_array_2
  >,
  AlgorithmDependencies<prefix_sum_scan_velo_track_hit_number_t,
    dev_velo_track_hit_number,
    dev_prefix_sum_auxiliary_array_2
  >,
  AlgorithmDependencies<consolidate_tracks_t,
    dev_atomics_storage,
    dev_tracks,
    dev_velo_track_hit_number,
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_velo_track_hits,
    dev_velo_states
  >,


  AlgorithmDependencies<velo_kalman_fit_t,
    dev_atomics_storage,
    dev_velo_track_hit_number,
    dev_velo_track_hits,
    dev_velo_states,
    dev_kalmanvelo_states
  >,

  AlgorithmDependencies<getSeeds_t,
    dev_kalmanvelo_states,
    dev_atomics_storage,
    dev_velo_track_hit_number,
    dev_seeds,
    dev_number_seeds
  >,


  AlgorithmDependencies<fitSeeds_t,
    dev_vertex,
    dev_number_vertex,
    dev_seeds,
    dev_number_seeds,
    dev_kalmanvelo_states,
    dev_atomics_storage,
    dev_velo_track_hit_number
  >,


  AlgorithmDependencies<ut_calculate_number_of_hits_t,
    dev_ut_raw_input,
    dev_ut_raw_input_offsets,
    dev_ut_hit_offsets
  >,
  AlgorithmDependencies<prefix_sum_reduce_ut_hits_t,
    dev_ut_hit_offsets,
    dev_prefix_sum_auxiliary_array_3
  >,
  AlgorithmDependencies<prefix_sum_single_block_ut_hits_t,
    dev_ut_hit_offsets,
    dev_prefix_sum_auxiliary_array_3
  >,
  AlgorithmDependencies<prefix_sum_scan_ut_hits_t,
    dev_ut_hit_offsets,
    dev_prefix_sum_auxiliary_array_3
  >,
  AlgorithmDependencies<ut_pre_decode_t,
    dev_ut_raw_input,
    dev_ut_raw_input_offsets,
    dev_ut_hits,
    dev_ut_hit_offsets,
    dev_ut_hit_count
  >,
  AlgorithmDependencies<ut_find_permutation_t,
    dev_ut_hits,
    dev_ut_hit_offsets,
    dev_ut_hit_permutations
  >,
  AlgorithmDependencies<ut_decode_raw_banks_in_order_t,
    dev_ut_raw_input,
    dev_ut_raw_input_offsets,
    dev_ut_hits,
    dev_ut_hit_offsets,
    dev_ut_hit_count,
    dev_ut_hit_permutations
  >,
  AlgorithmDependencies<veloUT_t,
    dev_ut_hits,
    dev_ut_hit_offsets,
    dev_atomics_storage,
    dev_velo_track_hit_number,
    dev_velo_track_hits,
    dev_velo_states,
    dev_veloUT_tracks,
    dev_atomics_veloUT
  >,
  AlgorithmDependencies<scifi_calculate_cluster_count_t,
    dev_scifi_raw_input,
    dev_scifi_raw_input_offsets,
    dev_scifi_hit_count
  >,
  AlgorithmDependencies<prefix_sum_reduce_scifi_hits_t,
    dev_scifi_hit_count,
    dev_prefix_sum_auxiliary_array_4
  >,
  AlgorithmDependencies<prefix_sum_single_block_scifi_hits_t,
    dev_scifi_hit_count,
    dev_prefix_sum_auxiliary_array_4
  >,
  AlgorithmDependencies<prefix_sum_scan_scifi_hits_t,
    dev_scifi_hit_count,
    dev_prefix_sum_auxiliary_array_4
  >,
  AlgorithmDependencies<scifi_pre_decode_t,
    dev_scifi_raw_input,
    dev_scifi_raw_input_offsets,
    dev_scifi_hit_count,
    dev_scifi_hits
  >,
  AlgorithmDependencies<scifi_raw_bank_decoder_t,
    dev_scifi_raw_input,
    dev_scifi_raw_input_offsets,
    dev_scifi_hit_count,
    dev_scifi_hits
  >,
  AlgorithmDependencies<scifi_pr_forward_t,
    dev_scifi_hits,
    dev_scifi_hit_count,
    dev_atomics_storage,
    dev_velo_track_hit_number,
    dev_velo_states,
    dev_veloUT_tracks,
    dev_atomics_veloUT,
    dev_scifi_tracks,
    dev_n_scifi_tracks
  >,
  AlgorithmDependencies<cpu_scifi_pr_forward_t,
    dev_scifi_hits,
    dev_scifi_hit_count
  >,
  AlgorithmDependencies<muon_catboost_features_extraction_t,
    dev_muon_track,
    dev_muon_hits,
    dev_muon_catboost_features
  >
> algorithms_dependencies_t;

/**
 * @brief Output arguments, ie. that cannot be freed.
 * @details The arguments specified in this type will
 *          be kept allocated since their first appearance
 *          until the end of the sequence.
 */
typedef std::tuple<
  dev_atomics_storage,
  dev_velo_track_hit_number,
  dev_velo_track_hits,
  dev_atomics_veloUT,
  dev_veloUT_tracks,
  dev_scifi_tracks,
  dev_n_scifi_tracks
> output_arguments_t;
