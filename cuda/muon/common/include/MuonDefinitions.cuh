#pragma once

using MuonTrackExtrapolation = std::vector<std::pair<float, float>>;
//using LHCbID = int;

namespace Muon {
  namespace Constants {
    /* Detector description
       There are four stations with number of regions in it
       in current implementation regions are ignored
    */
    static constexpr uint n_stations            = 4;
     /* Cut-offs */
    static constexpr uint max_numhits_per_event = 10000;
    static constexpr float SQRT3                = 1.7320508075688772;
    static constexpr float INVSQRT3             = 0.5773502691896258;
    static constexpr float MSFACTOR             = 5.552176750308537;
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

  struct State {
    State(float x, float y, float tx, float ty, float p) : x(x), y(y), tx(tx), ty(ty), p(p) {}
    float x;
    float y;
    float tx;
    float ty;
    float p;
    //float z;
  };
}
