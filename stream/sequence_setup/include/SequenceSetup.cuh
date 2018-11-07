#pragma once

#include <iostream>
#include "ArgumentEnum.cuh"
#include "ConfiguredSequence.cuh"
#include "TupleIndicesChecker.cuh"
#include "Argument.cuh"

// Prepared for C++17 variant
// template<typename T>
// auto transition(const T& state) {
//   return typename std::tuple_element<tuple_index<T, sequence_tuple_n>::value + 1, sequence_tuple_n>::type{};
// }

/**
 * @brief Argument tuple definition. All arguments and their types should
 *        be populated here. The order must be the same as arg_enum_t
 *        (checked at compile time).
 */
using argument_tuple_t = std::tuple<
  Argument<arg::dev_raw_input, char>,
  Argument<arg::dev_raw_input_offsets, uint>,
  Argument<arg::dev_estimated_input_size, uint>,
  Argument<arg::dev_module_cluster_num, uint>,
  Argument<arg::dev_module_candidate_num, uint>,
  Argument<arg::dev_cluster_offset, uint>,
  Argument<arg::dev_cluster_candidates, uint>,
  Argument<arg::dev_velo_cluster_container, uint>,
  Argument<arg::dev_tracks, Velo::TrackHits>,
  Argument<arg::dev_tracks_to_follow, uint>,
  Argument<arg::dev_hit_used, bool>,
  Argument<arg::dev_atomics_storage, int>,
  Argument<arg::dev_tracklets, Velo::TrackletHits>,
  Argument<arg::dev_weak_tracks, Velo::TrackletHits>,
  Argument<arg::dev_h0_candidates, short>,
  Argument<arg::dev_h2_candidates, short>,
  Argument<arg::dev_rel_indices, unsigned short>,
  Argument<arg::dev_hit_permutation, uint>,
  Argument<arg::dev_velo_track_hit_number, uint>,
  Argument<arg::dev_prefix_sum_auxiliary_array_2, uint>,
  Argument<arg::dev_velo_track_hits, uint>,
  Argument<arg::dev_velo_states, uint>,

  // TODO: check and try to use char instead of uint for dev_ut_raw_input variable
  // Changing uint to char cause a strange error:
  // Error: Internal Compiler Error (codegen): "there was an error in verifying the lgenfe output!"
  Argument<arg::dev_ut_raw_input, uint>,
  Argument<arg::dev_ut_raw_input_offsets, uint>,
  Argument<arg::dev_ut_hit_offsets, uint>,
  Argument<arg::dev_ut_hit_count, uint>,
  Argument<arg::dev_prefix_sum_auxiliary_array_3, uint>,
  Argument<arg::dev_ut_hits, uint>,
  Argument<arg::dev_ut_hit_permutations, uint>,
  Argument<arg::dev_veloUT_tracks, VeloUTTracking::TrackUT>,
  Argument<arg::dev_atomics_veloUT, int>,

  Argument<arg::dev_scifi_raw_input_offsets, uint>,
  Argument<arg::dev_scifi_hit_count, uint>,
  Argument<arg::dev_prefix_sum_auxiliary_array_4, uint>,
  Argument<arg::dev_scifi_hit_permutations, uint>,
  Argument<arg::dev_scifi_hits, uint>,
  Argument<arg::dev_scifi_raw_input, char>,
  Argument<arg::dev_scifi_tracks, SciFi::Track>,
  Argument<arg::dev_n_scifi_tracks, uint>
>;

/**
 * @brief Returns an array with names for every argument.
 */
std::array<std::string, std::tuple_size<argument_tuple_t>::value> get_argument_names();

/**
 * @brief Retrieves the sequence dependencies.
 * @details The sequence dependencies specifies for each algorithm
 *          in the sequence the datatypes it depends on from the arguments.
 *
 *          Note that this vector of arguments may vary from the actual
 *          arguments in the kernel invocation: ie. some cases:
 *          * if something is passed by value
 *          * if a pointer is set to point somewhere different from the beginning
 *          * if an argument is repeated in the argument list.
 */
std::vector<std::vector<int>> get_sequence_dependencies();

/**
 * @brief Retrieves the persistent datatypes.
 * @details The sequence may contain some datatypes that
 *          once they are reserved should never go out of memory (persistent).
 *          ie. the tracking sequence may produce some result and
 *          the primary vertex recostruction a different result.
 *          All output arguments should be returned here.
 */
std::vector<int> get_sequence_output_arguments();

/**
 * @brief Checks the argument tuple is defined sequentially and
 *        starting at 0.
 */
static_assert(check_tuple_indices<argument_tuple_t>(),
  "Argument tuple indices are not sequential starting at zero");
