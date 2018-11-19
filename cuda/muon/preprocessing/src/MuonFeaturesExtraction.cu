#include "MuonFeaturesExtraction.cuh"
#include <stdio.h>
__global__ void muon_catboost_features_extraction(
  const Muon::State* muTrack,
  const Muon::HitsSoA* muon_hits,
  float* dev_muon_catboost_features
) {
  // features
  //std::vector<double> times, dts, cross, resX, resY, minDist, distSeedHit;
  float times[Muon::Constants::n_stations];
  float dts[Muon::Constants::n_stations];
  float cross[Muon::Constants::n_stations];
  float resX[Muon::Constants::n_stations];
  float resY[Muon::Constants::n_stations];
  float minDist[Muon::Constants::n_stations];
  float distSeedHit[Muon::Constants::n_stations];
  float stationZ[Muon::Constants::n_stations];
  printf("PRINT FROM KERNEL\n");
  printf("%p\n", muon_hits);
  printf("%f\n", muon_hits->x[0]);
  printf("%f\n", muon_hits->station_offsets[0]);
  printf("%f %f %f %f %f\n", muTrack->x, muTrack->tx, muTrack->y, muTrack->ty, muTrack->p);
  printf("=================\n");
    /*
  //std::vector<float> stationZ;
  std::vector<std::pair<float,float>> extrapolation;
  for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
    const int station_offset = muon_hits->station_offsets[i_station];
    stationZ.push_back(muon_hits->z[station_offset]);
    extrapolation.emplace_back(
      muTrack->x + muTrack->tx * stationZ[i_station],
      muTrack->y + muTrack->ty * stationZ[i_station]
    );
  }
  
  for( unsigned int st = 0; st != Muon::Constants::n_stations; ++st ) {
    times.push_back(-10000.);
    dts.push_back(-10000.);
    cross.push_back(0.);
    resX.push_back(-10000.);
    resY.push_back(-10000.);
    minDist.push_back(1e10);
    distSeedHit.push_back(1e6);
  }
  std::vector<int> closestHits(Muon::Constants::n_stations);
  unsigned s = 0;
  for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
    const int station_offset = muon_hits->station_offsets[i_station];
    const int number_of_hits = muon_hits->number_of_hits_per_station[i_station];
    for(int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
      const int idx = station_offset + i_hit;
      const int id = muon_hits->tile[idx];
      s = i_station;
      distSeedHit[s] = (muon_hits->x[idx] - extrapolation[s].first) * (muon_hits->x[idx] - extrapolation[s].first) + 
                       (muon_hits->y[idx] - extrapolation[s].second) * (muon_hits->y[idx] - extrapolation[s].second);
      if(distSeedHit[s] < minDist[s]) {
        minDist[s] = distSeedHit[s];
        closestHits[s] = id;
      }
    }
  }
  
  float commonFactor = Muon::Constants::MSFACTOR/muTrack->p;
  for( unsigned int st = 0; st != Muon::Constants::n_stations; ++st ) {
    unsigned s = 0;
    int idFromTrack = closestHits[st];
    for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
      const int station_offset = muon_hits->station_offsets[i_station];
      const int number_of_hits = muon_hits->number_of_hits_per_station[i_station];
      for(int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
        const int idx = station_offset + i_hit;
        int idFromHit = muon_hits->tile[idx];
        if (idFromHit == idFromTrack) {
          s = i_station;
          times[s] = muon_hits->time[idx];
          dts[s] = muon_hits->delta_time[idx];
          (muon_hits->uncrossed[idx]==0) ? cross[s] = 2. : cross[s] = muon_hits->uncrossed[idx];
          float travDist = sqrt((stationZ[s]-stationZ[0]) * (stationZ[s]-stationZ[0]) +
                        (extrapolation[s].first-extrapolation[0].first) * (extrapolation[s].first-extrapolation[0].first) +
                        (extrapolation[s].second-extrapolation[0].second) * (extrapolation[s].second-extrapolation[0].second));
          float errMS = commonFactor*travDist*sqrt(travDist)*0.23850119787527452;
          if(std::abs(extrapolation[s].first-muon_hits->x[idx]) != 2000) {
            resX[s] = (extrapolation[s].first-muon_hits->x[idx]) / 
              sqrt((muon_hits->dx[idx] * Muon::Constants::INVSQRT3) * (muon_hits->dx[idx] * Muon::Constants::INVSQRT3) + errMS * errMS);
          }
          if(std::abs(extrapolation[s].second-muon_hits->y[idx]) != 2000) {
            resY[s] = (extrapolation[s].second-muon_hits->y[idx]) / 
              sqrt((muon_hits->dy[idx] * Muon::Constants::INVSQRT3) * (muon_hits->dy[idx] * Muon::Constants::INVSQRT3) + errMS * errMS);
          }
        }
      }
    }
  }
  
  for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
    dev_muon_catboost_features[i_station] = dts[i_station];
  }
  for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
    dev_muon_catboost_features[4 + i_station] = times[i_station];
  }
  for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
    dev_muon_catboost_features[8 + i_station] = cross[i_station];
  }
  for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
    dev_muon_catboost_features[12 + i_station] = resX[i_station];
  }
  for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
    dev_muon_catboost_features[16 + i_station] = resY[i_station];
  }*/
}
