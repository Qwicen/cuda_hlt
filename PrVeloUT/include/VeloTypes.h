#pragma once

#include <cmath>
#include <vector>

#include "../../main/include/CudaCommon.h"
#include "../../cuda/velo/common/include/VeloDefinitions.cuh"
#include "../../cuda/veloUT/common/include/VeloUTDefinitions.cuh"

namespace VeloUTTracking {

  struct Hit {
    
    float x;
    float z;
    
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
    // TODO we have this?
    // inline int planeCode() const { return 2 * ( m_cluster.station() - 1 ) + ( m_cluster.layer() - 1 ) % 2; }
    //inline int planeCode() const { return 2 * ( m_cluster_station - 1 ) + ( m_cluster_layer - 1 ) % 2; }
    inline int planeCode() const { return m_planeCode; }
    inline float sinT() const { return tanT() * cosT(); }
    // inline int size() const { return m_cluster.pseudoSize(); }
    inline float tanT() const { return -1 * dxDy(); }
    inline float weight2() const { return m_weight2; }
    inline float xAt( const float globalY ) const { return m_xAtYEq0 + globalY * dxDy(); }
    inline float xAtYEq0() const { return m_xAtYEq0; }
    //inline float xAtYMid() const { return m_x; }  // not used
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
    unsigned int hitsNum;
    std::vector<unsigned int> LHCbIDs;
    float qop;
    float quality;
    float chi2;
    std::vector<float> trackParams;

    void addLHCbID( unsigned int id ) {
      LHCbIDs.push_back(id);
      hitsNum++;
    }

    void set_qop( float _qop ) {
      qop = _qop;
    }
  };
  
  struct TrackVelo {
    VeloState state;
    TrackUT track;
  };

  struct TrackVeloUT {
    FullState state_beamline;
    FullState state_endvelo;
    FullState state_forward;
    TrackUT track;
    TrackUT trackForward;
  };
}
