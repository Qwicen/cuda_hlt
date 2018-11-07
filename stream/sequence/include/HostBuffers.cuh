#pragma once

#include "CudaCommon.h"
#include "VeloEventModel.cuh"
#include "UTDefinitions.cuh"
#include "SciFiDefinitions.cuh"

struct HostBuffers {
  // Pinned host datatypes
  uint* host_velo_tracks_atomics;
  uint* host_velo_track_hit_number;
  char* host_velo_track_hits;
  uint* host_total_number_of_velo_clusters;
  uint* host_number_of_reconstructed_velo_tracks;
  uint* host_accumulated_number_of_hits_in_velo_tracks;
  char* host_velo_states;
  uint* host_accumulated_number_of_ut_hits;
  SciFi::Track* host_scifi_tracks;
  uint* host_n_scifi_tracks;

  // UT tracking
  int* host_atomics_veloUT;
  VeloUTTracking::TrackUT* host_veloUT_tracks;

  // SciFi Decoding
  uint* host_accumulated_number_of_scifi_hits;

  /**
   * @brief Reserves all host buffers.
   */
  void reserve(const uint max_number_of_events);

  /**
   * @brief Returns total number of velo track hits.
   */
  size_t velo_track_hit_number_size();

  /**
   * @brief Retrieve total number of hit bytes.
   */
  uint32_t scifi_hits_bytes();
};
