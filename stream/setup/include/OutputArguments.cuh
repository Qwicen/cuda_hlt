#pragma once

#include "ArgumentsCommon.cuh"
#include "ArgumentsVelo.cuh"
#include "ArgumentsPV.cuh"
#include "ArgumentsUT.cuh"
#include "ArgumentsSciFi.cuh"
#include "ArgumentsMuon.cuh"

/**
 * @brief Output arguments, ie. that cannot be freed.
 * @details The arguments specified in this type will
 *          be kept allocated since their first appearance
 *          until the end of the sequence.
 */
typedef std::tuple<
  dev_atomics_velo,
  dev_velo_track_hit_number,
  dev_velo_track_hits,
  dev_atomics_ut,
  dev_ut_track_hit_number,
  dev_ut_track_hits,
  dev_ut_qop,
  dev_ut_track_velo_indices,
  dev_atomics_scifi,
  dev_scifi_track_hits,
  dev_scifi_track_hit_number,
  dev_scifi_qop,
  dev_scifi_states,
  dev_scifi_track_ut_indices,
  dev_event_list,
  dev_number_of_selected_events,
  dev_is_muon>
  output_arguments_t;
