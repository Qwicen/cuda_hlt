#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Common.h"
#include "Logger.h"
#include "VeloDefinitions.cuh"


#include "assert.h"

 struct FullState { 
    float x, y, tx, ty, qOverP = 0.;
    float c00, c11, c22, c33, c44, c10, c20, c30, c40, c21, c31, c41, c32, c42, c43 = 0.;
    float chi2 = 0.;
    float z = 0.;
  };

namespace ForwardTracking {

  
  /* Detector description
     There are three stations with four layers each 
  */
  static constexpr uint n_stations           = 3;
  static constexpr uint n_layers_per_station = 4;
  static constexpr uint n_layers             = 24;
  /* For now, the dxdy and planeCode are attributes of every hit,
     we should see if we cannot use these constants if we always know which plane a hit belongs to
     -> check how this is known in the final fitting step and which information
     needs to be saved for the forward tracking
  */
  // layer configuration: XUVX, U and V layers tilted by +/- 5 degrees = 0.087 radians
  // this is the same for each station
  static constexpr float dxDyTable[n_layers_per_station] = {0., 0.08748866617679595947, -0.08748866617679595947, 0.};
  static constexpr int layerCode[n_layers] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ,17, 18 ,19, 20, 21, 22, 23};
  
  /* Cut-offs */
  static constexpr uint max_numhits_per_layer = 2000;
  static constexpr uint max_numhits_per_event = 16000;
  static constexpr uint max_hit_candidates_per_layer = 200;

  /* SoA for hit variables
     The hits for every layer are written behind each other, the offsets 
     are stored for access;
     one Hits structure exists per event
   */
  struct HitsSoAFwd {
    int layer_offset[n_layers] = {0};
    
    float m_x[max_numhits_per_event] = {0}; 
    float m_z[max_numhits_per_event] = {0}; 
    float m_w[max_numhits_per_event] = {0};
    float m_dxdy[max_numhits_per_event] = {0};
    float m_dzdy[max_numhits_per_event] = {0};
    float m_yMin[max_numhits_per_event] = {0};
    float m_yMax[max_numhits_per_event] = {0};
    unsigned int m_LHCbID[max_numhits_per_event] = {0};
    int m_planeCode[max_numhits_per_event] = {0};
    int m_hitZone[max_numhits_per_event] = {0};
    bool m_used[max_numhits_per_event] = {false};
    // For Hough transform
    float m_coord[max_numhits_per_event] = {0};

    // check for used hit
    bool isValid( int value ) const {
      return !m_used[value];
    }

  };

  struct LineFitterPars {
    float   m_z0 = 0.; 
    float   m_c0 = 0.; 
    float   m_tc = 0.; 

    float m_s0 = 0.; 
    float m_sz = 0.; 
    float m_sz2 = 0.; 
    float m_sc = 0.; 
    float m_scz = 0.;   
  };

  struct TrackForward {

    std::vector< unsigned int > LHCbIDs;
    std::vector< unsigned int> hit_indices;
    float qop;
    unsigned short hitsNum = 0;
    float quality;
    float chi2;
    std::vector<float> trackParams;
    FullState state_endvelo;

    __host__  void addLHCbID( unsigned int id ) {
      //LHCbIDs[hitsNum++] = id;
      LHCbIDs.push_back( id );
      hitsNum = LHCbIDs.size();
    }
    
    __host__ void set_qop( float _qop ) {
      qop = _qop;
    }
  };
  
}
