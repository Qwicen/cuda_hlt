#pragma once

#include "Argument.cuh"
#include "VeloEventModel.cuh"
#include "UTDefinitions.cuh"
#include "PrForward.cuh"
#include "MuonDefinitions.cuh"
#include "patPV_Definitions.cuh"
#include "MiniState.cuh"

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

ARGUMENT(dev_pvtracks, PVTrack)
ARGUMENT(dev_zhisto, float)
ARGUMENT(dev_zpeaks, float)
ARGUMENT(dev_number_of_zpeaks, uint)
ARGUMENT(dev_multi_fit_vertices, PV::Vertex)
ARGUMENT(dev_number_of_multi_fit_vertices, uint)
ARGUMENT(dev_seeds, PatPV::XYZPoint)
ARGUMENT(dev_number_seeds, uint)
ARGUMENT(dev_vertex, PV::Vertex)
ARGUMENT(dev_number_vertex, int)

ARGUMENT(dev_accepted_velo_tracks, bool)
ARGUMENT(dev_velo_pv_ip, char)

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

ARGUMENT(dev_muon_hits, Muon::HitsSoA)
ARGUMENT(dev_muon_catboost_features, float)
ARGUMENT(dev_muon_catboost_output, float)

ARGUMENT(dev_kf_tracks, ParKalmanFilter::FittedTrack);
