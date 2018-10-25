#include "SequenceSetup.cuh"
#include "TupleTools.cuh"

std::array<std::string, std::tuple_size<argument_tuple_t>::value> get_argument_names() {
  std::array<std::string, std::tuple_size<argument_tuple_t>::value> a;
  a[arg::dev_raw_input] = "dev_raw_input";
  a[arg::dev_raw_input_offsets] = "dev_raw_input_offsets";
  a[arg::dev_estimated_input_size] = "dev_estimated_input_size";
  a[arg::dev_module_cluster_num] = "dev_module_cluster_num";
  a[arg::dev_module_candidate_num] = "dev_module_candidate_num";
  a[arg::dev_cluster_offset] = "dev_cluster_offset";
  a[arg::dev_cluster_candidates] = "dev_cluster_candidates";
  a[arg::dev_velo_cluster_container] = "dev_velo_cluster_container";
  a[arg::dev_tracks] = "dev_tracks";
  a[arg::dev_tracks_to_follow] = "dev_tracks_to_follow";
  a[arg::dev_hit_used] = "dev_hit_used";
  a[arg::dev_atomics_storage] = "dev_atomics_storage";
  a[arg::dev_tracklets] = "dev_tracklets";
  a[arg::dev_weak_tracks] = "dev_weak_tracks";
  a[arg::dev_h0_candidates] = "dev_h0_candidates";
  a[arg::dev_h2_candidates] = "dev_h2_candidates";
  a[arg::dev_rel_indices] = "dev_rel_indices";
  a[arg::dev_hit_permutation] = "dev_hit_permutation";
  a[arg::dev_velo_track_hit_number] = "dev_velo_track_hit_number";
  a[arg::dev_prefix_sum_auxiliary_array_2] = "dev_prefix_sum_auxiliary_array_2";
  a[arg::dev_velo_track_hits] = "dev_velo_track_hits";
  a[arg::dev_velo_states] = "dev_velo_states";
  a[arg::dev_ut_raw_input] = "dev_ut_raw_input";
  a[arg::dev_ut_raw_input_offsets] = "dev_ut_raw_input_offsets";
  a[arg::dev_ut_hit_offsets] = "dev_ut_hit_offsets";
  a[arg::dev_ut_hit_count] = "dev_ut_hit_count";
  a[arg::dev_ut_hits] = "dev_ut_hits";
  a[arg::dev_ut_hit_permutations] = "dev_ut_hit_permutations";
  a[arg::dev_veloUT_tracks] = "dev_veloUT_tracks";
  a[arg::dev_atomics_veloUT] = "dev_atomics_veloUT";
  a[arg::dev_scifi_raw_input] = "dev_scifi_raw_input";
  a[arg::dev_scifi_raw_input_offsets] = "dev_scifi_raw_input_offsets";
  a[arg::dev_scifi_hit_count] = "dev_scifi_hit_count";
  a[arg::dev_scifi_hit_permutations] = "dev_scifi_hit_permutations";
  a[arg::dev_prefix_sum_auxiliary_array_4] = "dev_prefix_sum_auxiliary_array_4";
  a[arg::dev_scifi_hits] = "dev_scifi_hits";
  a[arg::dev_scifi_tracks] = "dev_scifi_tracks";
  a[arg::dev_n_scifi_tracks] = "dev_n_scifi_tracks";
  return a;
}

std::vector<std::vector<int>> get_sequence_dependencies() {
  // Vector of dependecies for each algorithm
  // Note: It may be that some algorithms are not in the configured sequence_t.
  //       Regardless, all algorithms should have all dependencies specified below.
  //       A "hacky and easy" way to get around this problem is to specify the sequence_dependencies
  //       array to be of size tuple_size<sequence_t>::value + 1, as done below. That way, if the index is
  //       size+1 (meaning the type couldn't be found), the last element will be populated and
  //       possibly overriden if another algorithm is not in the sequence.
  //       The last element of sequence_dependencies should be considered garbage and not accessed for now.
  std::vector<std::vector<int>> sequence_dependencies (
    std::tuple_size<sequence_t>::value + 1
  );
 
  sequence_dependencies[tuple_contains<estimate_input_size_t, sequence_t>::index] = {
    arg::dev_raw_input,
    arg::dev_raw_input_offsets,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_module_candidate_num,
    arg::dev_cluster_candidates
  };
  sequence_dependencies[tuple_contains<prefix_sum_reduce_velo_clusters_t, sequence_t>::index] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_dependencies[tuple_contains<prefix_sum_single_block_velo_clusters_t, sequence_t>::index] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_dependencies[tuple_contains<prefix_sum_scan_velo_clusters_t, sequence_t>::index] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_dependencies[tuple_contains<masked_velo_clustering_t, sequence_t>::index] = {
    arg::dev_raw_input,
    arg::dev_raw_input_offsets,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_module_candidate_num,
    arg::dev_cluster_candidates,
    arg::dev_velo_cluster_container
  };
  sequence_dependencies[tuple_contains<calculate_phi_and_sort_t, sequence_t>::index] = {
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_velo_cluster_container,
    arg::dev_hit_permutation
  };
  sequence_dependencies[tuple_contains<fill_candidates_t, sequence_t>::index] = {
    arg::dev_velo_cluster_container,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_h0_candidates,
    arg::dev_h2_candidates
  };
  sequence_dependencies[tuple_contains<search_by_triplet_t, sequence_t>::index] = {
    arg::dev_velo_cluster_container,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_tracks,
    arg::dev_tracklets,
    arg::dev_tracks_to_follow,
    arg::dev_weak_tracks,
    arg::dev_hit_used,
    arg::dev_atomics_storage,
    arg::dev_h0_candidates,
    arg::dev_h2_candidates,
    arg::dev_rel_indices
  };
  sequence_dependencies[tuple_contains<weak_tracks_adder_t, sequence_t>::index] = {
    arg::dev_velo_cluster_container,
    arg::dev_estimated_input_size,
    arg::dev_tracks,
    arg::dev_weak_tracks,
    arg::dev_hit_used,
    arg::dev_atomics_storage
  };
  sequence_dependencies[tuple_contains<copy_and_prefix_sum_single_block_t, sequence_t>::index] = {
    arg::dev_atomics_storage
  };
  sequence_dependencies[tuple_contains<copy_velo_track_hit_number_t, sequence_t>::index] = {
    arg::dev_tracks,
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number
  };
  sequence_dependencies[tuple_contains<prefix_sum_reduce_velo_track_hit_number_t, sequence_t>::index] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_dependencies[tuple_contains<prefix_sum_single_block_velo_track_hit_number_t, sequence_t>::index] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_dependencies[tuple_contains<prefix_sum_scan_velo_track_hit_number_t, sequence_t>::index] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_dependencies[tuple_contains<consolidate_tracks_t, sequence_t>::index] = {
    arg::dev_atomics_storage,
    arg::dev_tracks,
    arg::dev_velo_track_hit_number,
    arg::dev_velo_cluster_container,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_velo_track_hits,
    arg::dev_velo_states
  };
  sequence_dependencies[tuple_contains<ut_calculate_number_of_hits_t, sequence_t>::index] = {
    arg::dev_ut_raw_input,
    arg::dev_ut_raw_input_offsets,
    arg::dev_ut_hit_offsets
  };
  sequence_dependencies[tuple_contains<prefix_sum_reduce_ut_hits_t, sequence_t>::index] = {
    arg::dev_ut_hit_offsets,
    arg::dev_prefix_sum_auxiliary_array_3
  };
  sequence_dependencies[tuple_contains<prefix_sum_single_block_ut_hits_t, sequence_t>::index] = {
    arg::dev_ut_hit_offsets,
    arg::dev_prefix_sum_auxiliary_array_3
  };
  sequence_dependencies[tuple_contains<prefix_sum_scan_ut_hits_t, sequence_t>::index] = {
    arg::dev_ut_hit_offsets,
    arg::dev_prefix_sum_auxiliary_array_3
  };
  sequence_dependencies[tuple_contains<ut_pre_decode_t, sequence_t>::index] = {
    arg::dev_ut_raw_input,
    arg::dev_ut_raw_input_offsets,
    arg::dev_ut_hits,
    arg::dev_ut_hit_offsets,
    arg::dev_ut_hit_count
  };
  sequence_dependencies[tuple_contains<ut_find_permutation_t, sequence_t>::index] = {
    arg::dev_ut_hits,
    arg::dev_ut_hit_offsets,
    arg::dev_ut_hit_permutations
  };
  sequence_dependencies[tuple_contains<ut_decode_raw_banks_in_order_t, sequence_t>::index] = {
    arg::dev_ut_raw_input,
    arg::dev_ut_raw_input_offsets,
    arg::dev_ut_hits,
    arg::dev_ut_hit_offsets,
    arg::dev_ut_hit_count,
    arg::dev_ut_hit_permutations
  };
  sequence_dependencies[tuple_contains<veloUT_t, sequence_t>::index] = {
    arg::dev_ut_hits,
    arg::dev_ut_hit_offsets,
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number,
    arg::dev_velo_track_hits,
    arg::dev_velo_states,
    arg::dev_veloUT_tracks,
    arg::dev_atomics_veloUT
  };
  sequence_dependencies[tuple_contains<estimate_cluster_count_t, sequence_t>::index] = {
    arg::dev_scifi_raw_input,
    arg::dev_scifi_raw_input_offsets,
    arg::dev_scifi_hit_count
  };
  sequence_dependencies[tuple_contains<prefix_sum_reduce_scifi_hits_t, sequence_t>::index] = {
    arg::dev_scifi_hit_count,
    arg::dev_prefix_sum_auxiliary_array_4
  };
  sequence_dependencies[tuple_contains<prefix_sum_single_block_scifi_hits_t, sequence_t>::index] = {
    arg::dev_scifi_hit_count,
    arg::dev_prefix_sum_auxiliary_array_4
  };
  sequence_dependencies[tuple_contains<prefix_sum_scan_scifi_hits_t, sequence_t>::index] = {
    arg::dev_scifi_hit_count,
    arg::dev_prefix_sum_auxiliary_array_4
  };
  sequence_dependencies[tuple_contains<raw_bank_decoder_t, sequence_t>::index] = {
    arg::dev_scifi_raw_input,
    arg::dev_scifi_raw_input_offsets,
    arg::dev_scifi_hit_count,
    arg::dev_scifi_hits
  };
  sequence_dependencies[tuple_contains<scifi_sort_by_x_t, sequence_t>::index] = {
    arg::dev_scifi_hits,
    arg::dev_scifi_hit_count,
    arg::dev_scifi_hit_permutations
  };
  sequence_dependencies[tuple_contains<scifi_pr_forward_t, sequence_t>::index] = {
    arg::dev_scifi_hits,
    arg::dev_scifi_hit_count,
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number,
    arg::dev_velo_states,
    arg::dev_veloUT_tracks,
    arg::dev_atomics_veloUT,
    arg::dev_scifi_tracks,
    arg::dev_n_scifi_tracks
  };

  return sequence_dependencies;
}

std::vector<int> get_sequence_output_arguments() {
  return {
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number,
    arg::dev_velo_track_hits,
    arg::dev_atomics_veloUT,
    arg::dev_veloUT_tracks,
    arg::dev_scifi_tracks,
    arg::dev_n_scifi_tracks
  };
}
