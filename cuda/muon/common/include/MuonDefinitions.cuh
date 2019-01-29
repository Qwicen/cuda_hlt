#pragma once

namespace Muon {
  namespace Constants {
    /* Detector description
       There are four stations with number of regions in it
       in current implementation regions are ignored
    */
    static constexpr uint n_stations            = 4;
     /* Cut-offs */
    static constexpr uint max_numhits_per_event = 200 * n_stations;
    static constexpr float SQRT3                = 1.7320508075688772;
    static constexpr float INVSQRT3             = 0.5773502691896258;
    static constexpr float MSFACTOR             = 5.552176750308537;

    /*Muon Catboost model uses 5 features for each station: Delta time, Time, Crossed, X residual, Y residual*/
    static constexpr uint n_catboost_features   = 5 * n_stations;
  }
  /* SoA for hit variables
    The hits for every layer are written behind each other, the offsets
    are stored for access;
    one Hits structure exists per event
  */
  struct HitsSoA {
    int number_of_hits_per_station[Constants::n_stations] = {0};
    int station_offsets[Constants::n_stations] = {0};
    int tile[Constants::max_numhits_per_event] = {0};
    float x[Constants::max_numhits_per_event] = {0};
    float dx[Constants::max_numhits_per_event] = {0};
    float y[Constants::max_numhits_per_event] = {0};
    float dy[Constants::max_numhits_per_event] = {0};
    float z[Constants::max_numhits_per_event] = {0};
    float dz[Constants::max_numhits_per_event] = {0};
    int uncrossed[Constants::max_numhits_per_event] = {0};
    unsigned int time[Constants::max_numhits_per_event] = {0};
    int delta_time[Constants::max_numhits_per_event] = {0};
    int cluster_size[Constants::max_numhits_per_event] = {0};

    static constexpr float eps = 1e-9;
    bool operator ==(const HitsSoA& other) {
      if (memcmp(number_of_hits_per_station, other.number_of_hits_per_station, sizeof(int) * Constants::n_stations) != 0) {
        return false;
      }
      if (memcmp(station_offsets, other.station_offsets, sizeof(int) * Constants::n_stations) != 0) {
        return false;
      }
      if (memcmp(tile, other.tile, sizeof(int) * Constants::max_numhits_per_event) != 0) {
        return false;
      }
      if (memcmp(uncrossed, other.uncrossed, sizeof(int) * Constants::max_numhits_per_event) != 0) {
        return false;
      }
      if (memcmp(time, other.time, sizeof(unsigned int) * Constants::max_numhits_per_event) != 0) {
        return false;
      }
      if (memcmp(delta_time, other.delta_time, sizeof(int) * Constants::max_numhits_per_event) != 0) {
        return false;
      }
      if (memcmp(cluster_size, other.cluster_size, sizeof(int) * Constants::max_numhits_per_event) != 0) {
        return false;
      }
      for (size_t i = 0; i < Constants::max_numhits_per_event; i++) {
        if (abs(x[i] - other.x[i]) >= eps ||
            abs(dx[i] - other.dx[i]) >= eps ||
            abs(y[i] - other.y[i]) >= eps ||
            abs(dy[i] - other.dy[i]) >= eps ||
            abs(z[i] - other.z[i]) >= eps ||
            abs(dz[i] - other.dz[i]) >= eps) {
          return false;
        }
      }
      return true;
    }
  };

  struct MuonCoords {
    int number_of_hits_per_station[Constants::n_stations] = {0};
    int station_offsets[Constants::n_stations] = {0};
    int digit_tile_station_offsets[Constants::n_stations] = {0};
    int number_of_digit_tiles_per_station[Constants::n_stations] = {0};
    int uncrossed[Constants::max_numhits_per_event] = {0};
    int digit_tile[Constants::max_numhits_per_event * 2] = {0};
    unsigned int digitTDC1[Constants::max_numhits_per_event] = {0};
    unsigned int digitTDC2[Constants::max_numhits_per_event] = {0};
  };
}
