#include "PrForwardWrapper.h"

void PrForwardWrapper(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  const Velo::Consolidated::States& velo_states,
  const uint event_tracks_offset,
  const UT::Consolidated::Tracks& ut_tracks,
  const int n_veloUT_tracks,
  const SciFi::Tracking::TMVA* tmva1,
  const SciFi::Tracking::TMVA* tmva2,
  const SciFi::Tracking::Arrays* constArrays,
  SciFi::TrackHits outputTracks[SciFi::Constants::max_tracks],
  uint* n_forward_tracks)
{
  // Loop over the veloUT input tracks
  for (int i_veloUT_track = 0; i_veloUT_track < n_veloUT_tracks; ++i_veloUT_track) {
    const float qop_ut = ut_tracks.qop[i_veloUT_track];

    const int i_velo_track = ut_tracks.velo_track[i_veloUT_track];
    const uint velo_states_index = event_tracks_offset + i_velo_track;
    const MiniState velo_state {velo_states, velo_states_index};

    find_forward_tracks(
      scifi_hits,
      scifi_hit_count,
      qop_ut,
      i_veloUT_track,
      outputTracks,
      n_forward_tracks,
      tmva1,
      tmva2,
      constArrays,
      velo_state);
  }
}
