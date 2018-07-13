#include "SequenceSetup.cuh" 

std::array<std::string, std::tuple_size<algorithm_tuple_t>::value> get_sequence_names() {
  std::array<std::string, std::tuple_size<algorithm_tuple_t>::value> a;
  a[seq::estimate_input_size] = "Estimate input size";
  a[seq::prefix_sum_reduce] = "Prefix sum reduce";
  a[seq::prefix_sum_single_block] = "Prefix sum single block";
  a[seq::prefix_sum_scan] = "Prefix sum scan";
  a[seq::masked_velo_clustering] = "Masked Velo clustering";
  a[seq::calculate_phi_and_sort] = "Calculate phi and sort";
  a[seq::search_by_triplet] = "Search by triplet";
  a[seq::copy_and_prefix_sum_single_block] = "Copy and prefix sum single block";
  a[seq::copy_velo_track_hit_number] = "Copy Velo track hit number";
  a[seq::prefix_sum_reduce_velo_track_hit_number] = "Prefix sum reduce (2)";
  a[seq::prefix_sum_single_block_velo_track_hit_number] = "Prefix sum single block (2)";
  a[seq::prefix_sum_scan_velo_track_hit_number] = "Prefix sum scan (2)";
  a[seq::consolidate_tracks] = "Consolidate tracks";
  a[seq::veloUT] = "VeloUT tracking";
  return a;
}

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
  a[arg::dev_ut_hits] = "dev_ut_hits";
  a[arg::dev_veloUT_tracks] = "dev_veloUT_tracks";
  return a;
}

std::vector<std::vector<int>> get_sequence_dependencies() {
  // Vector of dependecies for each algorithm
  std::vector<std::vector<int>> sequence_dependencies (
    std::tuple_size<argument_tuple_t>::value
  );

  sequence_dependencies[seq::estimate_input_size] = {
    arg::dev_raw_input,
    arg::dev_raw_input_offsets,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_module_candidate_num,
    arg::dev_cluster_candidates
  };
  sequence_dependencies[seq::prefix_sum_reduce] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_dependencies[seq::prefix_sum_single_block] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_dependencies[seq::prefix_sum_scan] = {
    arg::dev_estimated_input_size,
    arg::dev_cluster_offset
  };
  sequence_dependencies[seq::masked_velo_clustering] = {
    arg::dev_raw_input,
    arg::dev_raw_input_offsets,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_module_candidate_num,
    arg::dev_cluster_candidates,
    arg::dev_velo_cluster_container
  };
  sequence_dependencies[seq::calculate_phi_and_sort] = {
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_velo_cluster_container,
    arg::dev_hit_permutation
  };
  sequence_dependencies[seq::search_by_triplet] = {
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
  sequence_dependencies[seq::copy_and_prefix_sum_single_block] = {
    arg::dev_atomics_storage
  };
  sequence_dependencies[seq::copy_velo_track_hit_number] = {
    arg::dev_tracks,
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number
  };
  sequence_dependencies[seq::prefix_sum_reduce_velo_track_hit_number] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_dependencies[seq::prefix_sum_single_block_velo_track_hit_number] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_dependencies[seq::prefix_sum_scan_velo_track_hit_number] = {
    arg::dev_velo_track_hit_number,
    arg::dev_prefix_sum_auxiliary_array_2
  };
  sequence_dependencies[seq::consolidate_tracks] = {
    arg::dev_atomics_storage,
    arg::dev_tracks,
    arg::dev_velo_track_hit_number,
    arg::dev_velo_cluster_container,
    arg::dev_estimated_input_size,
    arg::dev_module_cluster_num,
    arg::dev_velo_track_hits,
    arg::dev_velo_states
  };
  sequence_dependencies[seq::veloUT] = {
    arg::dev_ut_hits,
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number,
    arg::dev_velo_track_hits,
    arg::dev_velo_states,
    arg::dev_veloUT_tracks
  };

  return sequence_dependencies;
}

std::vector<int> get_sequence_output_arguments() {
  return {
    arg::dev_atomics_storage,
    arg::dev_velo_track_hit_number,
    arg::dev_velo_track_hits
  };
}
