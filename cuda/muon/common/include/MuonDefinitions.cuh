#pragma once

namespace Muon {
  namespace Constants {
    /* Detector description
       There are four stations with number of regions in it
       in current implementation regions are ignored
    */
    static constexpr uint n_stations            = 4;
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
    struct FieldOfInterest {
      /* FOI = a + b * exp(-c * p) in both x and y directions */
      const float factor = 1.2;
      const float param_a_x[Constants::n_stations][Constants::n_regions] = {
        { 5.2, 3.6, 2.4, 2.4 },
        { 5.7, 4.4, 2.8, 2.3 },
        { 5.1, 3.1, 2.3, 2.1 },
        { 5.8, 3.4, 2.6, 2.8 }
      };
      const float param_a_y[Constants::n_stations][Constants::n_regions] = {
        { 3.3, 2.1, 1.7, 1.6 },
        { 3.6, 2.8, 1.9, 1.8 },
        { 4.4, 3.3, 2.2, 2.2 },
        { 4.8, 3.9, 2.6, 2.3 }
      };
      const float param_b_x[Constants::n_stations][Constants::n_regions] = {
        { 31, 28, 21, 17 },
        { 30, 31, 27, 22 },
        { 28, 33, 35, 47 },
        { 31, 39, 56, 151}
      };
      const float param_b_y[Constants::n_stations][Constants::n_regions] = {
        { 17, 15,  9,   5 },
        { 26, 25, 16,  15 },
        { 30, 49, 57,  92 },
        { 32, 55, 96, 166 }
      };
      const float param_c_x[Constants::n_stations][Constants::n_regions] = {
        { 0.06, 0.08, 0.10, 0.15 },
        { 0.04, 0.06, 0.09, 0.12 },
        { 0.08, 0.15, 0.23, 0.36 },
        { 0.07, 0.14, 0.24, 0.49 }
      };
      const float param_c_y[Constants::n_stations][Constants::n_regions] = {
        { 0.13, 0.19, 0.19, 0.24 },
        { 0.11, 0.19, 0.21, 0.32 },
        { 0.10, 0.22, 0.30, 0.52 },
        { 0.08, 0.20, 0.34, 0.52 }
      };
    };
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
