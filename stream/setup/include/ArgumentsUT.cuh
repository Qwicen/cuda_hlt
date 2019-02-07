#pragma once

#include "Argument.cuh"
#include "UTDefinitions.cuh"
#include "UTEventModel.cuh"

/**
 * @brief Definition of arguments. All arguments should be defined here,
 *        with their associated type.
 */
ARGUMENT(dev_ut_hit_offsets, uint)
ARGUMENT(dev_ut_hit_count, uint)
ARGUMENT(dev_prefix_sum_auxiliary_array_3, uint)
ARGUMENT(dev_ut_hits, uint)
ARGUMENT(dev_ut_hit_permutations, uint)
ARGUMENT(dev_ut_tracks, UT::TrackHits)
ARGUMENT(dev_atomics_ut, int)
ARGUMENT(dev_prefix_sum_auxiliary_array_5, uint)
ARGUMENT(dev_ut_windows_layers, short)
ARGUMENT(dev_ut_active_tracks, int)
ARGUMENT(dev_ut_track_hit_number, uint)
ARGUMENT(dev_ut_track_hits, char)
ARGUMENT(dev_ut_qop, float)
ARGUMENT(dev_ut_track_velo_indices, uint)
