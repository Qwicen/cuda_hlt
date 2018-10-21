#pragma once

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
  VeloUTTracking::TrackUT* host_veloUT_tracks;
  int* host_atomics_veloUT;

  /* UT DECODING */
  UTHits* host_ut_hits_decoded;

  // SciFi Decoding
  uint* host_accumulated_number_of_scifi_hits;
};
