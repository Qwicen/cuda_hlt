#include "PrepareTracks.h"
#include "ClusteringDefinitions.cuh"
#include "InputTools.h"
#include "MCParticle.h"
#include "SciFiConsolidated.cuh"
#include "SciFiDefinitions.cuh"
#include "TrackChecker.h"
#include "Tracks.h"
#include "UTConsolidated.cuh"
#include "UTDefinitions.cuh"
#include "VeloConsolidated.cuh"
#include "VeloEventModel.cuh"

std::vector<trackChecker::Tracks> prepareVeloTracks(
  const uint* track_atomics,
  const uint* track_hit_number,
  const char* track_hits,
  const uint number_of_events)
{
  /* Tracks to be checked, save in format for checker */
  std::vector<trackChecker::Tracks> checker_tracks; // all tracks from all events
  for (uint i_event = 0; i_event < number_of_events; i_event++) {
    trackChecker::Tracks tracks; // all tracks within one event

    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) track_atomics, (uint*) track_hit_number, i_event, number_of_events};
    const uint number_of_tracks_event = velo_tracks.number_of_tracks(i_event);

    for (uint i_track = 0; i_track < number_of_tracks_event; i_track++) {
      trackChecker::Track t;
      t.p = 0.f;

      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(i_track);
      Velo::Consolidated::Hits velo_track_hits = velo_tracks.get_hits((char*) track_hits, i_track);

      for (int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit) {
        t.addId(velo_track_hits.LHCbID[i_hit]);
      }
      tracks.push_back(t);
    } // tracks
    checker_tracks.emplace_back(tracks);
  }

  return checker_tracks;
}

std::vector<trackChecker::Tracks> prepareUTTracks(
  const uint* velo_track_atomics,
  const uint* velo_track_hit_number,
  const char* velo_track_hits,
  const int* ut_track_atomics,
  const uint* ut_track_hit_number,
  const char* ut_track_hits,
  const uint* ut_track_velo_indices,
  const float* ut_qop,
  const uint number_of_events)
{
  std::vector<trackChecker::Tracks> checker_tracks; // all tracks from all events
  for (uint i_event = 0; i_event < number_of_events; i_event++) {
    trackChecker::Tracks tracks; // all tracks within one event

    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) velo_track_atomics, (uint*) velo_track_hit_number, i_event, number_of_events};
    const UT::Consolidated::Tracks ut_tracks {(uint*) ut_track_atomics,
                                              (uint*) ut_track_hit_number,
                                              (float*) ut_qop,
                                              (uint*) ut_track_velo_indices,
                                              i_event,
                                              number_of_events};
    const uint number_of_tracks_event = ut_tracks.number_of_tracks(i_event);

    for (uint i_track = 0; i_track < number_of_tracks_event; i_track++) {
      trackChecker::Track t;

      // momentum
      const float qop = ut_tracks.qop[i_track];
      t.p = 1.f / std::abs(qop);
      // hits in UT
      const uint ut_track_number_of_hits = ut_tracks.number_of_hits(i_track);
      const UT::Consolidated::Hits track_hits_ut = ut_tracks.get_hits((char*) ut_track_hits, i_track);
      for (int i_hit = 0; i_hit < ut_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_ut.LHCbID[i_hit]);
      }
      // get index to corresponding velo track
      const int velo_track_index = ut_tracks.velo_track[i_track];
      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(velo_track_index);
      const Velo::Consolidated::Hits track_hits_velo = velo_tracks.get_hits((char*) velo_track_hits, velo_track_index);
      // hits in Velo
      for (int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_velo.LHCbID[i_hit]);
      }
      tracks.push_back(t);
    } // tracks
    checker_tracks.emplace_back(tracks);
  }

  return checker_tracks;
}

std::vector<trackChecker::Tracks> prepareSciFiTracks(
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
  const uint number_of_events)
{
  const SciFi::SciFiGeometry scifi_geom(scifi_geometry);
  std::vector<trackChecker::Tracks> checker_tracks; // all tracks from all events
  for (uint i_event = 0; i_event < number_of_events; i_event++) {
    trackChecker::Tracks tracks; // all tracks within one event

    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) velo_track_atomics, (uint*) velo_track_hit_number, i_event, number_of_events};
    const UT::Consolidated::Tracks ut_tracks {(uint*) ut_track_atomics,
                                              (uint*) ut_track_hit_number,
                                              (float*) ut_qop,
                                              (uint*) ut_track_velo_indices,
                                              i_event,
                                              number_of_events};

    SciFi::Consolidated::Tracks scifi_tracks {(uint*) scifi_track_atomics,
                                              (uint*) scifi_track_hit_number,
                                              (float*) scifi_qop,
                                              (MiniState*) scifi_states,
                                              (uint*) scifi_track_ut_indices,
                                              i_event,
                                              number_of_events};
    const uint number_of_tracks_event = scifi_tracks.number_of_tracks(i_event);

    for (uint i_track = 0; i_track < number_of_tracks_event; i_track++) {
      trackChecker::Track t;

      // momentum
      const float qop = scifi_tracks.qop[i_track];
      t.p = 1.f / std::abs(qop);

      // add SciFi hits
      const uint scifi_track_number_of_hits = scifi_tracks.number_of_hits(i_track);
      SciFi::Consolidated::Hits track_hits_scifi =
        scifi_tracks.get_hits((char*) scifi_track_hits, i_track, &scifi_geom, inv_clus_res.data());
      for (int i_hit = 0; i_hit < scifi_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_scifi.LHCbID(i_hit));
      }

      // add UT hits
      const uint UT_track_index = scifi_tracks.ut_track[i_track];
      const uint ut_track_number_of_hits = ut_tracks.number_of_hits(UT_track_index);
      const UT::Consolidated::Hits track_hits_ut = ut_tracks.get_hits((char*) ut_track_hits, UT_track_index);
      for (int i_hit = 0; i_hit < ut_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_ut.LHCbID[i_hit]);
      }

      // add Velo hits
      const int velo_track_index = ut_tracks.velo_track[UT_track_index];
      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(velo_track_index);
      const Velo::Consolidated::Hits track_hits_velo = velo_tracks.get_hits((char*) velo_track_hits, velo_track_index);
      for (int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_velo.LHCbID[i_hit]);
      }
      tracks.push_back(t);
    } // tracks
    checker_tracks.emplace_back(tracks);
  }

  return checker_tracks;
}
