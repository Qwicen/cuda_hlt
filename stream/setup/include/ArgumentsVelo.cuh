#pragma once

#include "Argument.cuh"
#include "VeloEventModel.cuh"

/**
 * @brief Definition of arguments. All arguments should be defined here,
 *        with their associated type.
 */
ARGUMENT(dev_estimated_input_size, uint)
ARGUMENT(dev_module_cluster_num, uint)
ARGUMENT(dev_module_candidate_num, uint)
ARGUMENT(dev_cluster_offset, uint)
ARGUMENT(dev_cluster_candidates, uint)
ARGUMENT(dev_velo_cluster_container, uint)
ARGUMENT(dev_tracks, Velo::TrackHits)
ARGUMENT(dev_tracks_to_follow, uint)
ARGUMENT(dev_hit_used, bool)
ARGUMENT(dev_atomics_velo, int)
ARGUMENT(dev_tracklets, Velo::TrackletHits)
ARGUMENT(dev_weak_tracks, Velo::TrackletHits)
ARGUMENT(dev_h0_candidates, short)
ARGUMENT(dev_h2_candidates, short)
ARGUMENT(dev_rel_indices, unsigned short)
ARGUMENT(dev_hit_permutation, uint)
ARGUMENT(dev_velo_track_hit_number, uint)
ARGUMENT(dev_prefix_sum_auxiliary_array_2, uint)
ARGUMENT(dev_velo_track_hits, char)
ARGUMENT(dev_velo_states, char)
ARGUMENT(dev_velo_kalman_beamline_states, char)
