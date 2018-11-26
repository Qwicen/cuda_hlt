#include "MuonFeaturesExtraction.cuh"
#include <stdio.h>
/*
dev_muon_catboost_features = 
    dts1,   dts2,   dts3,   dts4,
    times1, times2, times3, times4,
    cross1, cross2, cross3, cross4,
    resX1,  resX2,  resX3,  resX4,
    resY1,  resY2,  resY3,  resY4,
*/

enum offset {
  dts   = 0,
  times = 1 * Muon::Constants::n_stations,
  cross = 2 * Muon::Constants::n_stations,
  resX  = 3 * Muon::Constants::n_stations,
  resY  = 4 * Muon::Constants::n_stations
};

__global__ void muon_catboost_features_extraction(
  const Muon::State* muTrack,
  const Muon::HitsSoA* muon_hits,
  float* dev_muon_catboost_features
) {
  float minDist[Muon::Constants::n_stations];
  float distSeedHit[Muon::Constants::n_stations];
  float stationZ[Muon::Constants::n_stations];
  float extrapolation_x[Muon::Constants::n_stations];
  float extrapolation_y[Muon::Constants::n_stations];
  int closestHits[Muon::Constants::n_stations];

  for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
    const int station_offset = muon_hits->station_offsets[i_station];
    stationZ[i_station] = muon_hits->z[station_offset];
    minDist[i_station] = 1e10;
    extrapolation_x[i_station] = muTrack->x + muTrack->tx * stationZ[i_station];
    extrapolation_y[i_station] = muTrack->y + muTrack->ty * stationZ[i_station];
  }

  for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
    const int station_offset = muon_hits->station_offsets[i_station];
    const int number_of_hits = muon_hits->number_of_hits_per_station[i_station];
    for(int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
      const int idx = station_offset + i_hit;
      const int id = muon_hits->tile[idx];
      distSeedHit[i_station] = (muon_hits->x[idx] - extrapolation_x[i_station]) * (muon_hits->x[idx] - extrapolation_x[i_station]) + 
                       (muon_hits->y[idx] - extrapolation_y[i_station]) * (muon_hits->y[idx] - extrapolation_y[i_station]);
      if(distSeedHit[i_station] < minDist[i_station]) {
        minDist[i_station] = distSeedHit[i_station];
        closestHits[i_station] = id;
      }
    }
  }
  
  const float commonFactor = Muon::Constants::MSFACTOR/muTrack->p;
  for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
    const int idFromTrack = closestHits[i_station];
    const int station_offset = muon_hits->station_offsets[i_station];
    const int number_of_hits = muon_hits->number_of_hits_per_station[i_station];
    for(int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
      const int idx = station_offset + i_hit;
      const int idFromHit = muon_hits->tile[idx];
      if (idFromHit == idFromTrack) {
        
        dev_muon_catboost_features[offset::times + i_station] = muon_hits->time[idx];
        dev_muon_catboost_features[offset::dts + i_station] = muon_hits->delta_time[idx];
        const float cross = (muon_hits->uncrossed[idx]==0) ? 2. : muon_hits->uncrossed[idx];
        dev_muon_catboost_features[offset::cross + i_station] = cross;

        const float travDist = sqrt((stationZ[i_station]-stationZ[0]) * (stationZ[i_station]-stationZ[0]) +
                      (extrapolation_x[i_station]-extrapolation_x[0]) * (extrapolation_x[i_station]-extrapolation_x[0]) +
                      (extrapolation_y[i_station]-extrapolation_y[0]) * (extrapolation_y[i_station]-extrapolation_y[0]));
        const float errMS = commonFactor*travDist*sqrt(travDist)*0.23850119787527452;
        if(std::abs(extrapolation_x[i_station]-muon_hits->x[idx]) != 2000) {
          dev_muon_catboost_features[offset::resX + i_station] = (extrapolation_x[i_station]-muon_hits->x[idx]) / 
            sqrt((muon_hits->dx[idx] * Muon::Constants::INVSQRT3) * (muon_hits->dx[idx] * Muon::Constants::INVSQRT3) + errMS * errMS);
        }
        if(std::abs(extrapolation_y[i_station]-muon_hits->y[idx]) != 2000) {
          dev_muon_catboost_features[offset::resY + i_station] = (extrapolation_y[i_station]-muon_hits->y[idx]) / 
            sqrt((muon_hits->dy[idx] * Muon::Constants::INVSQRT3) * (muon_hits->dy[idx] * Muon::Constants::INVSQRT3) + errMS * errMS);
        }
      }
    }
  }
}
