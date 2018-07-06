#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../../../main/include/Common.h"

#include "assert.h"

namespace VeloUTTracking {

  
  /* Detector description
     There are two stations with two layers each 
  */
  static constexpr uint n_layers           = 4;
  static constexpr uint n_ut_hit_variables = 8;
  /* For now, the dxdy and planeCode are attributes of every hit,
     we should see if we cannot use these constants if we always know which plane a hit belongs to
     -> check how this is known in the final fitting step and which information
     needs to be saved for the forward tracking
  */
  // layer configuration: XUVX, U and V layers tilted by +/- 5 degrees
  static constexpr float dxdy[n_layers]    = {0., 0.087489, -0.087489, 0.}; 
  static constexpr int planeCode[n_layers] = {0, 1, 2, 3};
  
  /* Cut-offs */
  static constexpr uint max_numhits_per_layer = 500;
  static constexpr uint max_numhits_per_event = 4000; 

  /* SoA for hit variables
     The hits for every layer are written behind each other, the offsets 
     are stored for access;
     one Hits structure exists per event
   */
  struct HitsSoA {
    int layer_offset[n_layers];
    
    float m_cos[max_numhits_per_event];
    float m_yBegin[max_numhits_per_event];
    float m_yEnd[max_numhits_per_event];
    float m_dxDy[max_numhits_per_event];
    float m_zAtYEq0[max_numhits_per_event];
    float m_xAtYEq0[max_numhits_per_event];
    float m_weight2[max_numhits_per_event];
    int   m_highThreshold[max_numhits_per_event];
    unsigned int m_LHCbID[max_numhits_per_event];
    int m_planeCode[max_numhits_per_event];
    float x[max_numhits_per_event];
    float z[max_numhits_per_event];

    
    inline float cos(const int i_hit) const { return m_cos[i_hit]; }
    inline float cosT(const int i_hit) const { return ( std::fabs( m_xAtYEq0[i_hit] ) < 1.0E-9 ) ? 1. / std::sqrt( 1 + m_dxDy[i_hit] * m_dxDy[i_hit] ) : cos(i_hit); }
    inline float dxDy(const int i_hit) const { return m_dxDy[i_hit]; }
    inline bool highThreshold(const int i_hit) const { return m_highThreshold[i_hit]; }
    inline bool isYCompatible( const int i_hit, const float y, const float tol ) const { return yMin(i_hit) - tol <= y && y <= yMax(i_hit) + tol; }
    inline bool isNotYCompatible( const int i_hit, const float y, const float tol ) const { return yMin(i_hit) - tol > y || y > yMax(i_hit) + tol; }
    inline int lhcbID(const int i_hit) const { return m_LHCbID[i_hit]; }
    inline int planeCode(const int i_hit) const { return m_planeCode[i_hit]; }
    inline float sinT(const int i_hit) const { return tanT(i_hit) * cosT(i_hit); }
    inline float tanT(const int i_hit) const { return -m_dxDy[i_hit]; }
    inline float weight2(const int i_hit) const { return m_weight2[i_hit]; }
    inline float xAt( const int i_hit, const float globalY ) const { return m_xAtYEq0[i_hit] + globalY * m_dxDy[i_hit]; }
    inline float xAtYEq0(const int i_hit) const { return m_xAtYEq0[i_hit]; }
    //inline float xAtYMid(const int i_hit) const { return m_x[i_hit]; }  // not used, have to initialize properly if this  will be used
    inline float xMax(const int i_hit) const { return std::max( xAt( i_hit, yBegin(i_hit) ), xAt( i_hit, yEnd(i_hit) ) ); }
    inline float xMin(const int i_hit) const { return std::min( xAt( i_hit, yBegin(i_hit) ), xAt( i_hit, yEnd(i_hit) ) ); }
    inline float xT(const int i_hit) const { return cos(i_hit); }
    inline float yBegin(const int i_hit) const { return m_yBegin[i_hit]; }
    inline float yEnd(const int i_hit) const { return m_yEnd[i_hit]; }
    inline float yMax(const int i_hit) const { return std::max( yBegin(i_hit), yEnd(i_hit) ); }
    inline float yMid(const int i_hit) const { return 0.5 * ( yBegin(i_hit) + yEnd(i_hit) ); }
    inline float yMin(const int i_hit) const { return std::min( yBegin(i_hit), yEnd(i_hit) ); }
    inline float zAtYEq0(const int i_hit) const { return m_zAtYEq0[i_hit]; } 
    
    
  };



  





}
