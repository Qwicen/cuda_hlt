#pragma once

#include "../../../cuda/velo/calculate_phi_and_sort/include/CalculatePhiAndSort.cuh"
#include "../../../cuda/velo/consolidate_tracks/include/ConsolidateTracks.cuh"
#include "../../../cuda/velo/mask_clustering/include/MaskedVeloClustering.cuh"
#include "../../../cuda/velo/mask_clustering/include/EstimateInputSize.cuh"
#include "../../../cuda/velo/prefix_sum/include/PrefixSum.cuh"
#include "../../../cuda/velo/search_by_triplet/include/SearchByTriplet.cuh"
#include "../../../cuda/velo/simplified_kalman_filter/include/VeloKalmanFilter.cuh"
#include "../../handlers/include/HandlerMaker.cuh"
#include "../../gear/include/Argument.cuh"
#include "../../gear/include/Sequence.cuh"
#include "../../gear/include/TupleIndicesChecker.cuh"
#include "SequenceArgumentEnum.cuh"

/**
 * @brief Sequence tuple definition. All algorithms in the sequence
 *        should be added here in the same order as seq_enum_t (checked
 *        at compile time).
 *        
 *        make_handler receives as argument the kernel function itself and
 *        deduces its return type (void) and datatypes.
 */
using sequence_tuple_t = std::tuple<
  decltype(HandlerMaker<seq::estimate_input_size>::make_handler(estimate_input_size)),
  decltype(HandlerMaker<seq::prefix_sum_reduce>::make_handler(prefix_sum_reduce)),
  decltype(HandlerMaker<seq::prefix_sum_single_block>::make_handler(prefix_sum_single_block)),
  decltype(HandlerMaker<seq::prefix_sum_scan>::make_handler(prefix_sum_scan)),
  decltype(HandlerMaker<seq::masked_velo_clustering>::make_handler(masked_velo_clustering)),
  decltype(HandlerMaker<seq::calculate_phi_and_sort>::make_handler(calculatePhiAndSort)),
  decltype(HandlerMaker<seq::search_by_triplet>::make_handler(searchByTriplet)),
  decltype(HandlerMaker<seq::copy_and_prefix_sum_single_block>::make_handler(copy_and_prefix_sum_single_block)),
  decltype(HandlerMaker<seq::copy_velo_track_hit_number>::make_handler(copy_velo_track_hit_number)),
  decltype(HandlerMaker<seq::prefix_sum_reduce_velo_track_hit_number>::make_handler(prefix_sum_reduce)),
  decltype(HandlerMaker<seq::prefix_sum_single_block_velo_track_hit_number>::make_handler(prefix_sum_single_block)),
  decltype(HandlerMaker<seq::prefix_sum_scan_velo_track_hit_number>::make_handler(prefix_sum_scan)),
  decltype(HandlerMaker<seq::consolidate_tracks>::make_handler(consolidate_tracks))
>;

/**
 * Sequence type.
 */
using sequence_t = Sequence<sequence_tuple_t>;

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
  Argument<arg::dev_weak_tracks, uint>,
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
std::array<std::string, std::tuple_size<sequence_tuple_t>::value> get_sequence_names();

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
std::vector<std::vector<uint>> get_sequence_dependencies();

/**
 * @brief Checks the sequence tuple is defined sequentially and
 *        starting at 0.
 */
static_assert(check_tuple_indices<sequence_tuple_t>(),
  "Sequence tuple indices are not sequential starting at zero");

/**
 * @brief Checks the argument tuple is defined sequentially and
 *        starting at 0.
 */
static_assert(check_tuple_indices<argument_tuple_t>(),
  "Argument tuple indices are not sequential starting at zero");
