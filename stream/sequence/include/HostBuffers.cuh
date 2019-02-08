#pragma once

#include "CudaCommon.h"
#include "VeloEventModel.cuh"
#include "UTDefinitions.cuh"
#include "SciFiDefinitions.cuh"
#include "MuonDefinitions.cuh"
#include "TrackChecker.h"
#include "SciFiEventModel.cuh"
#include "UTEventModel.cuh"
#include "patPV_Definitions.cuh"
#include "PV_Definitions.cuh"
#include "ParKalmanDefinitions.cuh"

struct HostBuffers {
  // Pinned host datatypes
  uint* host_number_of_selected_events;
  uint* host_event_list;

  // Velo
  uint* host_atomics_velo;
  uint* host_velo_track_hit_number;
  char* host_velo_track_hits;
  uint* host_total_number_of_velo_clusters;
  uint* host_number_of_reconstructed_velo_tracks;
  uint* host_accumulated_number_of_hits_in_velo_tracks;
  char* host_kalmanvelo_states;
  PV::Vertex* host_reconstructed_pvs;
  int* host_number_of_vertex;
  int* host_number_of_seeds;
  uint* host_n_scifi_tracks;
  float* host_zhisto;
  float* host_peaks;
  uint* host_number_of_peaks;
  PV::Vertex* host_reconstructed_multi_pvs;
  int* host_number_of_multivertex;

  // UT
  int* host_atomics_ut;
  UT::TrackHits* host_ut_tracks;
  uint* host_number_of_reconstructed_ut_tracks;
  uint* host_accumulated_number_of_ut_hits;
  uint* host_accumulated_number_of_hits_in_ut_tracks;
  uint* host_ut_track_hit_number;
  char* host_ut_track_hits;
  float* host_ut_qop;
  uint* host_ut_track_velo_indices;

  // SciFi
  uint* host_accumulated_number_of_scifi_hits;
  uint* host_number_of_reconstructed_scifi_tracks;
  SciFi::TrackHits* host_scifi_tracks;
  int* host_atomics_scifi;
  uint* host_accumulated_number_of_hits_in_scifi_tracks;
  uint* host_scifi_track_hit_number;
  char* host_scifi_track_hits;
  float* host_scifi_qop;
  MiniState* host_scifi_states;
  uint* host_scifi_track_ut_indices;

  // Kalman
  ParKalmanFilter::FittedTrack* host_kf_tracks;

  // Non pinned datatypes: CPU algorithms
  std::vector<SciFi::TrackHits> scifi_tracks_events;
  std::vector<char> host_velo_states;

  /**
   * @brief Reserves all host buffers.
   */
  void reserve(const uint max_number_of_events);

  /**
   * @brief Returns total number of velo track hits.
   */
  size_t velo_track_hit_number_size() const;

  /**
   * @brief Returns total number of UT track hits.
   */
  size_t ut_track_hit_number_size() const;

  /**
   * @brief Returns total number of SciFi track hits.
   */
  size_t scifi_track_hit_number_size() const;

  /**
   * @brief Retrieve total number of hit uints.
   */
  uint32_t scifi_hits_uints() const;
};
