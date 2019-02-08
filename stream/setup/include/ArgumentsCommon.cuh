#pragma once

#include "Argument.cuh"

/**
 * @brief Definition of arguments. All arguments should be defined here,
 *        with their associated type.
 */
ARGUMENT(dev_velo_raw_input, char)
ARGUMENT(dev_velo_raw_input_offsets, uint)
ARGUMENT(dev_ut_raw_input, char)
ARGUMENT(dev_ut_raw_input_offsets, uint)
ARGUMENT(dev_scifi_raw_input, char)
ARGUMENT(dev_scifi_raw_input_offsets, uint)
ARGUMENT(dev_event_list, uint)
ARGUMENT(dev_event_order, uint)
ARGUMENT(dev_number_of_selected_events, uint)
ARGUMENT(dev_velo_pv_ip, char)
ARGUMENT(dev_accepted_velo_tracks, bool)
