#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../../../main/include/Common.h"
#include "../../../../main/include/Logger.h"
#include "../../../velo/common/include/VeloDefinitions.cuh"

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
  // layer configuration: XUVX, U and V layers tilted by +/- 5 degrees = 0.087 radians
  static constexpr float dxDyTable[n_layers] = {0., 0.08748867, -0.08748867, 0.};
  static constexpr int planeCode[n_layers] = {0, 1, 2, 3};
  
  /* Cut-offs */
  static constexpr uint max_numhits_per_layer = 500;
  static constexpr uint max_numhits_per_event = 4000;
  static constexpr uint max_hit_candidates_per_layer = 40;
  static constexpr uint max_num_tracks = 300; // to do: what is the best / safest value here?
  static constexpr uint max_track_size = VeloTracking::max_track_size + 8; // to do: double check what the max # of hits added in UT really is

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
    float m_zAtYEq0[max_numhits_per_event];
    float m_xAtYEq0[max_numhits_per_event];
    float m_weight2[max_numhits_per_event];
    int   m_highThreshold[max_numhits_per_event];
    unsigned int m_LHCbID[max_numhits_per_event];
    int m_planeCode[max_numhits_per_event];
    float x[max_numhits_per_event]; // calculated during VeloUT tracking
    float z[max_numhits_per_event]; // calculated during VeloUT tracking

    
    inline float cos(const int i_hit) const { return m_cos[i_hit]; }
    inline int planeCode(const int i_hit) const { return m_planeCode[i_hit]; }
    inline float dxDy(const int i_hit) const {
      const int i_plane = m_planeCode[i_hit];
      return dxDyTable[i_plane];
    }
    inline float cosT(const int i_hit) const { return ( std::fabs( m_xAtYEq0[i_hit] ) < 1.0E-9 ) ? 1. / std::sqrt( 1 + dxDy(i_hit) * dxDy(i_hit) ) : cos(i_hit); }
    inline bool highThreshold(const int i_hit) const { return m_highThreshold[i_hit]; }
    inline bool isYCompatible( const int i_hit, const float y, const float tol ) const { return yMin(i_hit) - tol <= y && y <= yMax(i_hit) + tol; }
    inline bool isNotYCompatible( const int i_hit, const float y, const float tol ) const { return yMin(i_hit) - tol > y || y > yMax(i_hit) + tol; }
    inline int LHCbID(const int i_hit) const { return m_LHCbID[i_hit]; }
    inline float sinT(const int i_hit) const { return tanT(i_hit) * cosT(i_hit); }
    inline float tanT(const int i_hit) const { return -1 * dxDy(i_hit); }
    inline float weight2(const int i_hit) const { return m_weight2[i_hit]; }
    inline float xAt( const int i_hit, const float globalY ) const { return m_xAtYEq0[i_hit] + globalY * dxDy(i_hit); }
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

  struct Hit {
    
    float x; // calculated during VeloUT tracking
    float z; // calculated during VeloUT tracking
    
    float m_cos;     
    float m_weight2;  ///< The hit weight^2 (1/error^2)
    float m_xAtYEq0; ///< The value of x at the point y=0
    float m_yBegin;  ///< The y value at the start point of the line
    float m_yEnd;    ///< The y value at the end point of the line
    float m_zAtYEq0; ///< The value of z at the point y=0
    unsigned int m_LHCbID;
    int m_planeCode;
    
    bool  m_highThreshold;
    
    Hit(){}
    
    inline float cos() const { return m_cos; }
    inline float dxDy() const { return dxDyTable[m_planeCode]; }
    inline float cosT() const { return ( std::fabs( m_xAtYEq0 ) < 1.0E-9 ) ? 1. / std::sqrt( 1 + dxDy() * dxDy() ) : cos(); }
    inline bool highThreshold() const { return m_highThreshold; }
    inline bool isYCompatible( const float y, const float tol ) const { return yMin() - tol <= y && y <= yMax() + tol; }
    inline bool isNotYCompatible( const float y, const float tol ) const { return yMin() - tol > y || y > yMax() + tol; }
    inline int LHCbID() const { return m_LHCbID; }
    inline int planeCode() const { return m_planeCode; }
    inline float sinT() const { return tanT() * cosT(); }
    inline float tanT() const { return -1 * dxDy(); }
    inline float weight2() const { return m_weight2; }
    inline float xAt( const float globalY ) const { return m_xAtYEq0 + globalY * dxDy(); }
    inline float xAtYEq0() const { return m_xAtYEq0; }
    inline float xMax() const { return std::max( xAt( yBegin() ), xAt( yEnd() ) ); }
    inline float xMin() const { return std::min( xAt( yBegin() ), xAt( yEnd() ) ); }
    inline float xT() const { return cos(); }
    inline float yBegin() const { return m_yBegin; }
    inline float yEnd() const { return m_yEnd; }
    inline float yMax() const { return std::max( yBegin(), yEnd() ); }
    inline float yMid() const { return 0.5 * ( yBegin() + yEnd() ); }
    inline float yMin() const { return std::min( yBegin(), yEnd() ); }
    inline float zAtYEq0() const { return m_zAtYEq0; }
  };

  Hit createHit( HitsSoA *hits_layers, const int i_layer, const int i_hit );
  
  typedef std::vector<Hit> Hits;

  struct TrackUT {
    
    unsigned int LHCbIDs[VeloUTTracking::max_track_size];
    float qop;
    unsigned short hitsNum = 0;
    
    
    void addLHCbID( unsigned int id ) {
      LHCbIDs[hitsNum++] = id;
    }

    void set_qop( float _qop ) {
      qop = _qop;
    }
  };
   
  /* Structure containing indices to hits within hit array */
  struct TrackHits { 
    unsigned short hitsNum = 0;
    unsigned short hits[VeloUTTracking::max_track_size];
    unsigned short veloTrackIndex;
    float qop;

    void addHit( const unsigned short _h ) {
      hits[hitsNum] = _h;
      ++hitsNum;
    }
    void set_qop( float _qop ) {
      qop = _qop;
    }
  };


  
}
