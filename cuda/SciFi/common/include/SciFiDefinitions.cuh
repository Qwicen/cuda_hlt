#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Common.h"
#include "Logger.h"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "VeloUTDefinitions.cuh"

#include "assert.h"

 struct FullState { 
    float x, y, tx, ty, qOverP = 0.;
    float c00, c11, c22, c33, c44, c10, c20, c30, c40, c21, c31, c41, c32, c42, c43 = 0.;
    float chi2 = 0.;
    float z = 0.;
  };

namespace SciFi {
  namespace Constants {
    /* Detector description
       There are three stations with four layers each 
    */
    static constexpr uint n_stations           = 3;
    static constexpr uint n_layers_per_station = 4;
    static constexpr uint n_zones              = 24;
    static constexpr uint n_layers             = 12;
        
    static constexpr int layerCode[n_zones] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ,17, 18 ,19, 20, 21, 22, 23};
    
    /* Cut-offs */
    static constexpr uint max_numhits_per_layer = 2000;
    static constexpr uint max_numhits_per_event = 16000;
    static constexpr uint max_hit_candidates_per_layer = 200;

  } // Constants
  
  /* SoA for hit variables
     The hits for every layer are written behind each other, the offsets 
     are stored for access;
     one Hits structure exists per event
  */
    struct HitsSoA {
      int layer_offset[Constants::n_zones] = {0};
      
      float m_x[Constants::max_numhits_per_event] = {0}; 
      float m_z[Constants::max_numhits_per_event] = {0}; 
      float m_w[Constants::max_numhits_per_event] = {0};
      float m_dxdy[Constants::max_numhits_per_event] = {0};
      float m_dzdy[Constants::max_numhits_per_event] = {0};
      float m_yMin[Constants::max_numhits_per_event] = {0};
      float m_yMax[Constants::max_numhits_per_event] = {0};
      unsigned int m_LHCbID[Constants::max_numhits_per_event] = {0};
      int m_planeCode[Constants::max_numhits_per_event] = {0};
      int m_hitZone[Constants::max_numhits_per_event] = {0};
           
    };
    
  const int max_tracks = 150;
  const int max_track_size = 15 + VeloUTTracking::max_track_size;
  
  struct Track {
    
    unsigned int LHCbIDs[max_track_size];
    float qop;
    unsigned short hitsNum = 0;
    float chi2;
    
    __host__  __device__ void addLHCbID( unsigned int id ) {
      assert( hitsNum < max_track_size - 1 );
      LHCbIDs[hitsNum++] =  id;
  }
    
    __host__ __device__ void set_qop( float _qop ) {
      qop = _qop;
    }
  };
  
} // SciFi
