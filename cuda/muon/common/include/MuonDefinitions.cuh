#pragma once

namespace Muon {
  namespace Constants {
    /* Detector description
       There are four stations with number of regions in it
       in current implementation regions are ignored
    */
    static constexpr uint n_stations           = 4;
    
    /* Cut-offs */
    static constexpr uint max_numhits_per_event = 10000; 
  } // Constants
  
    /* SoA for hit variables
       The hits for every layer are written behind each other, the offsets 
       are stored for access;
       one Hits structure exists per event
    */
    struct HitsSoA {
      int m_number_of_hits_per_station[Constants::n_stations] = {0};
      int m_station_offsets[Constants::n_stations] = {0};
      int m_tile[Constants::max_numhits_per_event] = {0}; 
      float m_x[Constants::max_numhits_per_event] = {0}; 
      float m_dx[Constants::max_numhits_per_event] = {0}; 
      float m_y[Constants::max_numhits_per_event] = {0}; 
      float m_dy[Constants::max_numhits_per_event] = {0}; 
      float m_z[Constants::max_numhits_per_event] = {0}; 
      float m_dz[Constants::max_numhits_per_event] = {0}; 
      int m_uncrossed[Constants::max_numhits_per_event] = {0};
      unsigned int m_time[Constants::max_numhits_per_event] = {0};
      int m_delta_time[Constants::max_numhits_per_event] = {0};
      int m_cluster_size[Constants::max_numhits_per_event] = {0};
           
    };
} // Muon
