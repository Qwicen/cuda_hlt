#include "PrepareKalmanTracks.h"

float ipKalman(const ParKalmanFilter::FittedTrack& track, const PV::Vertex& vertex)
{
  // Get position information.
  float tx = track.state[2];
  float ty = track.state[3];
  float dz = vertex.position.z - track.z;
  float dx = track.state[0] + dz * tx - vertex.position.x;
  float dy = track.state[1] + dz * ty - vertex.position.y;
  return std::sqrt((dx * dx + dy * dy) / (1.0f + tx * tx + ty * ty));
}

float ipxKalman(const ParKalmanFilter::FittedTrack& track, const PV::Vertex& vertex)
{
  // Get position information.
  float tx = track.state[2];
  float dz = vertex.position.z - track.z;
  float dx = track.state[0] + dz * tx - vertex.position.x;
  return dx;
}

float ipyKalman(const ParKalmanFilter::FittedTrack& track, const PV::Vertex& vertex)
{
  // Get position information.
  float ty = track.state[3];
  float dz = vertex.position.z - track.z;
  float dy = track.state[1] + dz * ty - vertex.position.y;
  return dy;
}

float ipChi2Kalman(const ParKalmanFilter::FittedTrack& track, const PV::Vertex& vertex)
{
  // Get position information.
  float tx = track.state[2];
  float ty = track.state[3];
  float dz = vertex.position.z - track.z;
  float dx = track.state[0] + dz * tx - vertex.position.x;
  float dy = track.state[1] + dz * ty - vertex.position.y;

  // Build covariance matrix.
  float cov00 = vertex.cov00 + track.cov(0, 0);
  float cov10 = vertex.cov10;
  float cov11 = vertex.cov11 + track.cov(1, 1);

  // Add contribution from extrapolation.
  cov00 += dz * dz * track.cov(2, 2) + 2 * dz * track.cov(2, 0);
  cov11 += dz * dz * track.cov(3, 3) + 2 * dz * track.cov(3, 1);

  // Add the contribution from the PV z position.
  cov00 += tx * tx * vertex.cov22 - 2 * tx * vertex.cov20;
  cov10 += tx * ty * vertex.cov22 - ty * vertex.cov20 - tx * vertex.cov21;
  cov11 += ty * ty * vertex.cov22 - 2 * ty * vertex.cov21;

  // Invert the covariance matrix.
  float D = cov00 * cov11 - cov10 * cov10;
  float invcov00 = cov11 / D;
  float invcov10 = -cov10 / D;
  float invcov11 = cov00 / D;

  return dx * dx * invcov00 + 2 * dx * dy * invcov10 + dy * dy * invcov11;
}

float ipVelo(const Velo::Consolidated::States& velo_kalman_states, const uint state_index, const PV::Vertex& vertex)
{
  // ORIGIN: Rec/Tr/TrackKernel/src/TrackVertexUtils.cpp
  float tx = velo_kalman_states.tx[state_index];
  float ty = velo_kalman_states.ty[state_index];
  float dz = vertex.position.z - velo_kalman_states.z[state_index];
  float dx = velo_kalman_states.x[state_index] + dz * tx - vertex.position.x;
  float dy = velo_kalman_states.y[state_index] + dz * ty - vertex.position.y;
  return std::sqrt((dx * dx + dy * dy) / (1.0f + tx * tx + ty * ty));
}

float ipxVelo(const Velo::Consolidated::States& velo_kalman_states, const uint state_index, const PV::Vertex& vertex)
{
  // ORIGIN: Rec/Tr/TrackKernel/src/TrackVertexUtils.cpp
  float tx = velo_kalman_states.tx[state_index];
  float dz = vertex.position.z - velo_kalman_states.z[state_index];
  float dx = velo_kalman_states.x[state_index] + dz * tx - vertex.position.x;
  return dx;
}

float ipyVelo(const Velo::Consolidated::States& velo_kalman_states, const uint state_index, const PV::Vertex& vertex)
{
  // ORIGIN: Rec/Tr/TrackKernel/src/TrackVertexUtils.cpp
  float ty = velo_kalman_states.ty[state_index];
  float dz = vertex.position.z - velo_kalman_states.z[state_index];
  float dy = velo_kalman_states.y[state_index] + dz * ty - vertex.position.y;
  return dy;
}

float ipChi2Velo(const Velo::Consolidated::States& velo_kalman_states, const uint state_index, const PV::Vertex& vertex)
{
  // ORIGIN: Rec/Tr/TrackKernel/src/TrackVertexUtils.cpp
  float tx = velo_kalman_states.tx[state_index];
  float ty = velo_kalman_states.ty[state_index];
  float dz = vertex.position.z - velo_kalman_states.z[state_index];
  float dx = velo_kalman_states.x[state_index] + dz * tx - vertex.position.x;
  float dy = velo_kalman_states.y[state_index] + dz * ty - vertex.position.y;

  // compute the covariance matrix. first only the trivial parts:
  float cov00 = vertex.cov00 + velo_kalman_states.c00[state_index];
  float cov10 = vertex.cov10; // state c10 is 0.f;
  float cov11 = vertex.cov11 + velo_kalman_states.c11[state_index];

  // add the contribution from the extrapolation
  cov00 += dz * dz * velo_kalman_states.c22[state_index] + 2 * std::abs(dz * velo_kalman_states.c20[state_index]);
  // cov10 is unchanged: state c32, c30 and c21 are  0.f
  cov11 += dz * dz * velo_kalman_states.c33[state_index] + 2 * dz * velo_kalman_states.c31[state_index];

  // add the contribution from pv Z
  cov00 += tx * tx * vertex.cov22 - 2 * tx * vertex.cov20;
  cov10 += tx * ty * vertex.cov22 - ty * vertex.cov20 - tx * vertex.cov21;
  cov11 += ty * ty * vertex.cov22 - 2 * ty * vertex.cov21;

  // invert the covariance matrix
  float D = cov00 * cov11 - cov10 * cov10;
  float invcov00 = cov11 / D;
  float invcov10 = -cov10 / D;
  float invcov11 = cov00 / D;

  return dx * dx * invcov00 + 2 * dx * dy * invcov10 + dy * dy * invcov11;
}

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
  char* velo_states_base,
  PV::Vertex* rec_vertex,
  const int* number_of_vertex,
  const uint number_of_events)
{

  const SciFi::SciFiGeometry scifi_geom(scifi_geometry);
  std::vector<trackChecker::Tracks> checker_tracks;

  // Loop over events.
  for (uint i_event = 0; i_event < number_of_events; i_event++) {
    trackChecker::Tracks tracks; // All tracks from one event.

    // Get the vertices.
    std::vector<PV::Vertex*> vecOfVertices;
    for (uint i = 0; i < number_of_vertex[i_event]; ++i) {
      int index = i_event * PatPV::max_number_vertices + i;
      vecOfVertices.push_back(&(rec_vertex[index]));
    }

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

    // Make the VELO states.
    const uint event_velo_tracks_offset = velo_tracks.tracks_offset(i_event);
    const Velo::Consolidated::States velo_states {velo_states_base, velo_tracks.total_number_of_tracks};

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

      // Calculate IP.
      t.kalman_ip_chi2 = 9999.;
      t.velo_ip_chi2 = 9999.;
      for (auto vertex : vecOfVertices) {
        float locIPChi2 = ipChi2Kalman(track, *vertex);
        if (locIPChi2 < t.kalman_ip_chi2) {
          t.kalman_ip = ipKalman(track, *vertex);
          t.kalman_ip_chi2 = locIPChi2;
          t.kalman_ipx = ipxKalman(track, *vertex);
          t.kalman_ipy = ipyKalman(track, *vertex);
        }
        locIPChi2 = ipChi2Velo(velo_states, event_velo_tracks_offset + velo_track_index, *vertex);
        if (locIPChi2 < t.velo_ip_chi2) {
          t.velo_ip = ipVelo(velo_states, event_velo_tracks_offset + velo_track_index, *vertex);
          t.velo_ip_chi2 = locIPChi2;
          t.velo_ipx = ipxVelo(velo_states, event_velo_tracks_offset + velo_track_index, *vertex);
          t.velo_ipy = ipyVelo(velo_states, event_velo_tracks_offset + velo_track_index, *vertex);
        }
      }

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
      t.first_qop = (float) track.first_qop;
      t.best_qop = (float) track.best_qop;

      tracks.push_back(t);

    } // Track loop.

    checker_tracks.emplace_back(tracks);

  } // Event loop.

  return checker_tracks;
}
