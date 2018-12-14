#include "MuonFeaturesExtraction.cuh"
#include "ConsolidateSciFi.cuh"
__global__ void muon_catboost_features_extraction(
  int* dev_atomics_scifi,
  uint* dev_scifi_track_hit_number,
  float* dev_scifi_qop,
  MiniState* dev_scifi_states,
  uint* dev_scifi_track_ut_indices,
  const Muon::HitsSoA* muon_hits,
  float* dev_muon_catboost_features
) {
  const uint number_of_events = gridDim.x;
  const uint event_id = blockIdx.x;
  const uint station_id = blockIdx.y;

  SciFi::Consolidated::Tracks scifi_tracks {
    (uint*)dev_atomics_scifi,
      dev_scifi_track_hit_number,
      dev_scifi_qop,
      dev_scifi_states,
      dev_scifi_track_ut_indices,
      event_id,
      number_of_events
  };

  const uint number_of_tracks_event = scifi_tracks.number_of_tracks(event_id);
  const uint event_offset = scifi_tracks.tracks_offset(event_id);
  for (uint track_id = threadIdx.x; track_id < number_of_tracks_event; track_id += blockDim.x) {
    float min_dist = 1e10;
    int index_of_closest_hit;

    const int station_offset = muon_hits[event_id].station_offsets[station_id];
    const int number_of_hits = muon_hits[event_id].number_of_hits_per_station[station_id];
    const float station_z = muon_hits[event_id].z[station_offset];
    const float station_z0 = muon_hits[event_id].z[muon_hits[event_id].station_offsets[0]];

    const float extrapolation_x = scifi_tracks.states[track_id].x + scifi_tracks.states[track_id].tx * (station_z - scifi_tracks.states[track_id].z);
    const float extrapolation_y = scifi_tracks.states[track_id].y + scifi_tracks.states[track_id].ty * (station_z - scifi_tracks.states[track_id].z);
    const float extrapolation_x0 = scifi_tracks.states[track_id].x + scifi_tracks.states[track_id].tx * (station_z0 - scifi_tracks.states[track_id].z);
    const float extrapolation_y0 = scifi_tracks.states[track_id].y + scifi_tracks.states[track_id].ty * (station_z0 - scifi_tracks.states[track_id].z);

    for (int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
      const int idx = station_offset + i_hit;

      const float hit_track_distance = (muon_hits[event_id].x[idx] - extrapolation_x) * (muon_hits[event_id].x[idx] - extrapolation_x) +
                      (muon_hits[event_id].y[idx] - extrapolation_y) * (muon_hits[event_id].y[idx] - extrapolation_y);

      if (hit_track_distance < min_dist) {
        //todo: possible mutex lock
        min_dist = hit_track_distance;
        index_of_closest_hit = i_hit;
      }
    }

    const uint tracks_features_offset = (event_offset + track_id) * Muon::Constants::n_catboost_features;
    const float common_factor = Muon::Constants::MSFACTOR * std::abs(scifi_tracks.qop[track_id]);

    const int idx = station_offset + index_of_closest_hit;

    dev_muon_catboost_features[tracks_features_offset + offset::TIMES + station_id] = muon_hits[event_id].time[idx];
    dev_muon_catboost_features[tracks_features_offset + offset::DTS + station_id] = muon_hits[event_id].delta_time[idx];
    dev_muon_catboost_features[tracks_features_offset + offset::CROSS + station_id] = (muon_hits[event_id].uncrossed[idx] == 0) ? 2. : muon_hits[event_id].uncrossed[idx];

    const float trav_dist = sqrt(
      (station_z - station_z0) * (station_z - station_z0) +
      (extrapolation_x - extrapolation_x0) * (extrapolation_x - extrapolation_x0) +
      (extrapolation_y - extrapolation_y0) * (extrapolation_y - extrapolation_y0)
    );
    const float errMS = common_factor * trav_dist * sqrt(trav_dist) * 0.23850119787527452;
  
    /*if (std::abs(extrapolation_x - muon_hits[event_id].x[idx]) != 2000)*/
    dev_muon_catboost_features[tracks_features_offset + offset::RES_X + station_id] = (extrapolation_x - muon_hits[event_id].x[idx]) /
      sqrt(
        (muon_hits[event_id].dx[idx] * Muon::Constants::INVSQRT3) *
        (muon_hits[event_id].dx[idx] * Muon::Constants::INVSQRT3) + errMS * errMS
      );
    /*if (std::abs(extrapolation_y - muon_hits[event_id].y[idx]) != 2000)*/
    dev_muon_catboost_features[tracks_features_offset + offset::RES_Y + station_id] = (extrapolation_y - muon_hits[event_id].y[idx]) /
      sqrt(
        (muon_hits[event_id].dy[idx] * Muon::Constants::INVSQRT3) *
        (muon_hits[event_id].dy[idx] * Muon::Constants::INVSQRT3) + errMS * errMS
      );
  }
}
