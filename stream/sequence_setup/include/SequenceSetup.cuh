#pragma once

#include "../../../cuda/velo/calculate_phi_and_sort/include/CalculatePhiAndSort.cuh"
#include "../../../cuda/velo/consolidate_tracks/include/ConsolidateTracks.cuh"
#include "../../../cuda/velo/mask_clustering/include/MaskedVeloClustering.cuh"
#include "../../../cuda/velo/mask_clustering/include/EstimateInputSize.cuh"
#include "../../../cuda/velo/prefix_sum/include/PrefixSum.cuh"
#include "../../../cuda/velo/search_by_triplet/include/SearchByTriplet.cuh"
#include "../../../cuda/velo/simplified_kalman_filter/include/VeloKalmanFilter.cuh"
#include "../../gear/include/Argument.cuh"
#include "../../gear/include/Sequence.cuh"
#include "../../gear/include/TupleIndicesChecker.cuh"
#include "SequenceArgumentEnum.cuh"

/**
 * @brief Algorithm tuple definition. All algorithms in the sequence
 *        should be added here in the same order as seq_enum_t
 *        (this condition is checked at compile time).
 */
constexpr auto sequence_algorithms() {
  return std::make_tuple(
    estimate_input_size,
    prefix_sum_reduce,
    prefix_sum_single_block,
    prefix_sum_scan,
    masked_velo_clustering,
    calculate_phi_and_sort,
    search_by_triplet,
    copy_and_prefix_sum_single_block,
    copy_velo_track_hit_number,
    prefix_sum_reduce,
    prefix_sum_single_block,
    prefix_sum_scan,
    consolidate_tracks
  );
}

/**
 * @brief Definition of the algorithm tuple type.
 *        make_algorithm_tuple receives as argument a tuple
 *        with the kernel functions and
 *        deduces its return type (void) and datatypes.
 */
using algorithm_tuple_t = decltype(make_algorithm_tuple(sequence_algorithms()));

/**
 * Sequence type.
 */
using sequence_t = Sequence<algorithm_tuple_t>;

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
  Argument<arg::dev_tracks, TrackHits>,
  Argument<arg::dev_tracks_to_follow, uint>,
  Argument<arg::dev_hit_used, bool>,
  Argument<arg::dev_atomics_storage, int>,
  Argument<arg::dev_tracklets, TrackHits>,
  Argument<arg::dev_weak_tracks, TrackHits>,
  Argument<arg::dev_h0_candidates, short>,
  Argument<arg::dev_h2_candidates, short>,
  Argument<arg::dev_rel_indices, unsigned short>,
  Argument<arg::dev_hit_permutation, uint>,
  Argument<arg::dev_velo_track_hit_number, uint>,
  Argument<arg::dev_prefix_sum_auxiliary_array_2, uint>,
  Argument<arg::dev_velo_track_hits, Hit<mc_check_enabled>>,
  Argument<arg::dev_velo_states, VeloState>
>;

/**
 * @brief Returns an array with names for every element in the sequence.
 */
std::array<std::string, std::tuple_size<algorithm_tuple_t>::value> get_sequence_names();

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
 * @brief Checks the sequence tuple is defined sequentially and
 *        starting at 0.
 */
static_assert(check_tuple_indices<algorithm_tuple_t>(),
  "Sequence tuple indices are not sequential starting at zero");

/**
 * @brief Checks the argument tuple is defined sequentially and
 *        starting at 0.
 */
static_assert(check_tuple_indices<argument_tuple_t>(),
  "Argument tuple indices are not sequential starting at zero");
