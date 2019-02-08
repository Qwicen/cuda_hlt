#include "PrepareKalmanTracks.h"

std::vector<trackChecker::Tracks> prepareKalmanTracks(
  const uint* velo_track_atomics,
  const uint* velo_track_hit_number,
  const char* velo_track_hits,
  const int* ut_track_atomics,
  const uint* ut_track_hit_number,
  const char* ut_track_hits,
  const uint* ut_track_velo_indices,
  const float* ut_qop,
  const int* scifi_track_atomics,
  const uint* scifi_track_hit_number,
  const char* scifi_track_hits,
  const uint* scifi_track_ut_indices,
  const float* scifi_qop,
  const MiniState* scifi_states,
  const char* scifi_geometry,
  const std::array<float, 9>& inv_clus_res,
  const ParKalmanFilter::FittedTrack* kf_tracks,
  const uint number_of_events)
{

  const SciFi::SciFiGeometry scifi_geom(scifi_geometry);
  std::vector<trackChecker::Tracks> checker_tracks;

  // Loop over events.
  for (uint i_event = 0; i_event < number_of_events; i_event++) {
    trackChecker::Tracks tracks; // All tracks from one event.

    // Make the consolidated tracks.
    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) velo_track_atomics, (uint*) velo_track_hit_number, i_event, number_of_events};
    const UT::Consolidated::Tracks ut_tracks {(uint*) ut_track_atomics,
                                              (uint*) ut_track_hit_number,
                                              (float*) ut_qop,
                                              (uint*) ut_track_velo_indices,
                                              i_event,
                                              number_of_events};
    const SciFi::Consolidated::Tracks scifi_tracks {(uint*) scifi_track_atomics,
                                                    (uint*) scifi_track_hit_number,
                                                    (float*) scifi_qop,
                                                    (MiniState*) scifi_states,
                                                    (uint*) scifi_track_ut_indices,
                                                    i_event,
                                                    number_of_events};

    // Loop over tracks.
    const uint number_of_tracks_event = scifi_tracks.number_of_tracks(i_event);
    for (uint i_track = 0; i_track < number_of_tracks_event; i_track++) {
      trackChecker::Track t;

      // Add SciFi hits.
      const uint scifi_track_number_of_hits = scifi_tracks.number_of_hits(i_track);
      SciFi::Consolidated::Hits track_hits_scifi =
        scifi_tracks.get_hits((char*) scifi_track_hits, i_track, &scifi_geom, inv_clus_res.data());
      for (int i_hit = 0; i_hit < scifi_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_scifi.LHCbID(i_hit));
      }

      // Add UT hits.
      const uint UT_track_index = scifi_tracks.ut_track[i_track];
      const uint ut_track_number_of_hits = ut_tracks.number_of_hits(UT_track_index);
      const UT::Consolidated::Hits track_hits_ut = ut_tracks.get_hits((char*) ut_track_hits, UT_track_index);
      for (int i_hit = 0; i_hit < ut_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_ut.LHCbID[i_hit]);
      }

      // Add Velo hits.
      const int velo_track_index = ut_tracks.velo_track[UT_track_index];
      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(velo_track_index);
      const Velo::Consolidated::Hits track_hits_velo = velo_tracks.get_hits((char*) velo_track_hits, velo_track_index);
      for (int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_velo.LHCbID[i_hit]);
      }

      ParKalmanFilter::FittedTrack track = kf_tracks[scifi_tracks.tracks_offset(i_event) + i_track];

      // Get kalman filter information.
      t.z = (float) track.z;
      t.x = (float) track.state[0];
      t.y = (float) track.state[1];
      t.tx = (float) track.state[2];
      t.ty = (float) track.state[3];
      t.qop = (float) track.state[4];
      t.chi2 = (float) track.chi2;
      t.chi2V = (float) track.chi2V;
      t.chi2T = (float) track.chi2T;
      t.ndof = track.ndof;
      t.ndofV = track.ndofV;
      t.ndofT = track.ndofT;

      tracks.push_back(t);

    } // Track loop.

    checker_tracks.emplace_back(tracks);

  } // Event loop.

  return checker_tracks;
}
