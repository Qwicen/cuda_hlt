#pragma once

namespace Muon {
  namespace Constants {
    /* Detector description
       There are four stations with number of regions in it
       in current implementation regions are ignored
    */
    static constexpr uint n_regions             = 4;
    /* Cut-offs */
    static constexpr uint max_numhits_per_event = 200 * n_stations;
    static constexpr float SQRT3                = 1.7320508075688772;
    static constexpr float INVSQRT3             = 0.5773502691896258;
    static constexpr float MSFACTOR             = 5.552176750308537;

    /*Muon Catboost model uses 5 features for each station: Delta time, Time, Crossed, X residual, Y residual*/
    static constexpr uint n_catboost_features   = 5 * n_stations;

    /* IsMuon constants */
    static constexpr float momentum_cuts[]      = {3000, 6000, 10000};
    namespace FOI {
      using namespace xy = std::pair<float, float>;
      static constexpr float factor = 1.2;
      static constexpr std::vector<xy> a = {
        { xy(5.2, 3.3), xy(3.6, 2.1), xy(2.4, 1.7), xy(2.4, 1.6) },
        { xy(5.7, 3.6), xy(4.4, 2.8), xy(2.8, 1.9), xy(2.3, 1.8) },
        { xy(5.1, 4.4), xy(3.1, 3.3), xy(2.3, 2.2), xy(2.1, 2.2) },
        { xy(5.8, 4.8), xy(3.4, 3.9), xy(2.6, 2.6), xy(2.8, 2.3) }
      };
      static constexpr std::vector<xy> b = {
        { xy(31, 17), xy(28, 15), xy(21,  9), xy(17, 5) },
        { xy(30, 26), xy(31, 25), xy(27, 16), xy(22, 15) },
        { xy(28, 30), xy(33, 49), xy(35, 57), xy(47, 92) },
        { xy(31, 32), xy(39, 55), xy(56, 96), xy(151,166) }
      };
      static constexpr std::vector<xy> c = {
        { xy(0.06, 0.13), xy(0.08, 0.19), xy(0.10, 0.19), xy(0.15, 0.24) },
        { xy(0.04, 0.11), xy(0.06, 0.19), xy(0.09, 0.21), xy(0.12, 0.32) },
        { xy(0.08, 0.10), xy(0.15, 0.22), xy(0.23, 0.30), xy(0.36, 0.52) },
        { xy(0.07, 0.08), xy(0.14, 0.20), xy(0.24, 0.34), xy(0.49, 0.52) }
      };
    }
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
  };
}
