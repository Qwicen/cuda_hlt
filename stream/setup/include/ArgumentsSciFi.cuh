#pragma once

#include "Argument.cuh"
#include "SciFiEventModel.cuh"
#include "MiniState.cuh"

/**
 * @brief Definition of arguments. All arguments should be defined here,
 *        with their associated type.
 */
ARGUMENT(dev_scifi_hit_count, uint)
ARGUMENT(dev_prefix_sum_auxiliary_array_4, uint)
ARGUMENT(dev_scifi_hit_permutations, uint)
ARGUMENT(dev_scifi_hits, uint)
ARGUMENT(dev_scifi_tracks, SciFi::TrackHits)
ARGUMENT(dev_atomics_scifi, int)
ARGUMENT(dev_prefix_sum_auxiliary_array_6, uint)
ARGUMENT(dev_scifi_track_hit_number, uint)
ARGUMENT(dev_scifi_track_hits, char)
ARGUMENT(dev_scifi_qop, float)
ARGUMENT(dev_scifi_states, MiniState)
ARGUMENT(dev_scifi_track_ut_indices, uint)
