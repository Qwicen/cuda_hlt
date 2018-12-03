#include "MuonFeaturesExtraction.cuh"

enum offset {
  DTS   = 0,
  TIMES = 1 * Muon::Constants::n_stations,
  CROSS = 2 * Muon::Constants::n_stations,
  RES_X = 3 * Muon::Constants::n_stations,
  RES_Y = 4 * Muon::Constants::n_stations
};

__global__ void muon_catboost_features_extraction(
  const MiniState* mu_track,
  const Muon::HitsSoA* muon_hits,
  const float* scifi_qop,
  float* dev_muon_catboost_features
) {
  const int i_station = blockIdx.x;

  float dist_seed_hit;
  float min_dist = 1e10;
  int closest_hits;
  
  const int station_offset = muon_hits->station_offsets[i_station];
  const int number_of_hits = muon_hits->number_of_hits_per_station[i_station];
  const float station_z = muon_hits->z[station_offset];
  const float station_z0 = muon_hits->z[muon_hits->station_offsets[0]];
  const float extrapolation_x = mu_track->x + mu_track->tx * station_z;
  const float extrapolation_y = mu_track->y + mu_track->ty * station_z;
  const float extrapolation_x0 = mu_track->x + mu_track->tx * station_z0;
  const float extrapolation_y0 = mu_track->y + mu_track->ty * station_z0;

  for(int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
    const int idx = station_offset + i_hit;
    const int id = muon_hits->tile[idx];
    
    dist_seed_hit = (muon_hits->x[idx] - extrapolation_x) * (muon_hits->x[idx] - extrapolation_x) + 
                    (muon_hits->y[idx] - extrapolation_y) * (muon_hits->y[idx] - extrapolation_y);

    if(dist_seed_hit < min_dist) {
      min_dist = dist_seed_hit;
      closest_hits = id;
    }
  }
  
  const float common_factor = Muon::Constants::MSFACTOR*std::abs(scifi_qop[0]); //todo:block id
  for(int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
    const int idx = station_offset + i_hit;
    if (muon_hits->tile[idx] == closest_hits) {
      dev_muon_catboost_features[offset::TIMES + i_station] = muon_hits->time[idx];
      dev_muon_catboost_features[offset::DTS + i_station] = muon_hits->delta_time[idx];
      dev_muon_catboost_features[offset::CROSS + i_station] = (muon_hits->uncrossed[idx]==0) ? 2. : muon_hits->uncrossed[idx];

      const float trav_dist = sqrt(
                    (station_z - station_z0) * (station_z - station_z0) +
                    (extrapolation_x - extrapolation_x0) * (extrapolation_x - extrapolation_x0) +
                    (extrapolation_y - extrapolation_y0) * (extrapolation_y - extrapolation_y0)
                  );
      const float errMS = common_factor * trav_dist * sqrt(trav_dist) * 0.23850119787527452;

      if(std::abs(extrapolation_x - muon_hits->x[idx]) != 2000) {
        dev_muon_catboost_features[offset::RES_X + i_station] = (extrapolation_x-muon_hits->x[idx]) / 
          sqrt(
            (muon_hits->dx[idx] * Muon::Constants::INVSQRT3) * 
            (muon_hits->dx[idx] * Muon::Constants::INVSQRT3) + errMS * errMS
          );
      }
      if(std::abs(extrapolation_y - muon_hits->y[idx]) != 2000) {
        dev_muon_catboost_features[offset::RES_Y + i_station] = (extrapolation_y-muon_hits->y[idx]) / 
          sqrt(
            (muon_hits->dy[idx] * Muon::Constants::INVSQRT3) * 
            (muon_hits->dy[idx] * Muon::Constants::INVSQRT3) + errMS * errMS
          );
      }
    }
  }
}
