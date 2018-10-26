#pragma once

#include <tuple>
#include "Argument.cuh"
#include "VeloEventModel.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrForward.cuh"

/**
 * @brief Definition of arguments. All arguments should be defined here,
 *        with their associated type.
 */
ARGUMENT(dev_raw_input, char)
ARGUMENT(dev_raw_input_offsets, uint)
ARGUMENT(dev_estimated_input_size, uint)
ARGUMENT(dev_module_cluster_num, uint)
ARGUMENT(dev_module_candidate_num, uint)
ARGUMENT(dev_cluster_offset, uint)
ARGUMENT(dev_cluster_candidates, uint)
ARGUMENT(dev_velo_cluster_container, uint)
ARGUMENT(dev_tracks, Velo::TrackHits)
ARGUMENT(dev_tracks_to_follow, uint)
ARGUMENT(dev_hit_used, bool)
ARGUMENT(dev_atomics_storage, int)
ARGUMENT(dev_tracklets, Velo::TrackletHits)
ARGUMENT(dev_weak_tracks, Velo::TrackletHits)
ARGUMENT(dev_h0_candidates, short)
ARGUMENT(dev_h2_candidates, short)
ARGUMENT(dev_rel_indices, unsigned short)
ARGUMENT(dev_hit_permutation, uint)
ARGUMENT(dev_velo_track_hit_number, uint)
ARGUMENT(dev_prefix_sum_auxiliary_array_2, uint)
ARGUMENT(dev_velo_track_hits, uint)
ARGUMENT(dev_velo_states, uint)
ARGUMENT(dev_ut_raw_input, uint)
ARGUMENT(dev_ut_raw_input_offsets, uint)
ARGUMENT(dev_ut_hit_offsets, uint)
ARGUMENT(dev_ut_hit_count, uint)
ARGUMENT(dev_prefix_sum_auxiliary_array_3, uint)
ARGUMENT(dev_ut_hits, uint)
ARGUMENT(dev_ut_hit_permutations, uint)
ARGUMENT(dev_veloUT_tracks, VeloUTTracking::TrackUT)
ARGUMENT(dev_atomics_veloUT, int)
ARGUMENT(dev_scifi_raw_input_offsets, uint)
ARGUMENT(dev_scifi_hit_count, uint)
ARGUMENT(dev_prefix_sum_auxiliary_array_4, uint)
ARGUMENT(dev_scifi_hit_permutations, uint)
ARGUMENT(dev_scifi_hits, char)
ARGUMENT(dev_scifi_raw_input, char)
ARGUMENT(dev_scifi_tracks, Scifi::Track)
ARGUMENT(dev_n_scifi_tracks, uint)

/**
 * @brief Argument tuple definition. All arguments should be added here.
 *        Unfortunately, I didn't figure out a way to do that automagically.
 */
using argument_tuple_t = std::tuple<
  dev_raw_input,
  dev_raw_input_offsets,
  dev_estimated_input_size,
  dev_module_cluster_num,
  dev_module_candidate_num,
  dev_cluster_offset,
  dev_cluster_candidates,
  dev_velo_cluster_container,
  dev_tracks,
  dev_tracks_to_follow,
  dev_hit_used,
  dev_atomics_storage,
  dev_tracklets,
  dev_weak_tracks,
  dev_h0_candidates,
  dev_h2_candidates,
  dev_rel_indices,
  dev_hit_permutation,
  dev_velo_track_hit_number,
  dev_prefix_sum_auxiliary_array_2,
  dev_velo_track_hits,
  dev_velo_states,
  dev_ut_raw_input,
  dev_ut_raw_input_offsets,
  dev_ut_hit_offsets,
  dev_ut_hit_count,
  dev_prefix_sum_auxiliary_array_3,
  dev_ut_hits,
  dev_ut_hit_permutations,
  dev_veloUT_tracks,
  dev_atomics_veloUT,
  dev_scifi_raw_input_offsets,
  dev_scifi_hit_count,
  dev_prefix_sum_auxiliary_array_4,
  dev_scifi_hit_permutations,
  dev_scifi_hits,
  dev_scifi_raw_input,
  dev_scifi_tracks,
  dev_n_scifi_tracks
>;
