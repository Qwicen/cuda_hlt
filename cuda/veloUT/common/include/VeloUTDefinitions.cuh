#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Common.h"
#include "VeloDefinitions.cuh"

#include "assert.h"


namespace VeloUTTracking {

  static constexpr int num_atomics = 2;
  
  /* Detector description
     There are two stations with two layers each 
  */
  static constexpr uint n_layers           = 4;
  static constexpr uint n_ut_hit_variables = 8;
  
  /* Cut-offs */
  static constexpr uint max_numhits_per_layer = 500;
  static constexpr uint max_numhits_per_event = 6000;
  static constexpr uint max_hit_candidates_per_layer = 100;
  static constexpr uint max_num_tracks = 400; // to do: what is the best / safest value here?
  static constexpr uint max_track_size = VeloTracking::max_track_size + 8; // to do: double check what the max # of hits added in UT really is

  /* SoA for hit variables
     The hits for every layer are written behind each other, the offsets 
     are stored for access;
     one Hits structure exists per event
   */
  struct HitsSoA {
    int layer_offset[n_layers];
    int n_hits_layers[n_layers];
    
    float m_yBegin[max_numhits_per_event];
    float m_yEnd[max_numhits_per_event];
    float m_zAtYEq0[max_numhits_per_event];
    float m_xAtYEq0[max_numhits_per_event];
    float m_weight[max_numhits_per_event];
    int   m_highThreshold[max_numhits_per_event];
    unsigned int m_LHCbID[max_numhits_per_event];
    int m_planeCode[max_numhits_per_event];

    
    __host__ __device__ inline int planeCode(const int i_hit) const { return m_planeCode[i_hit]; }
    __host__ __device__ inline float cosT(const int i_hit, const float dxDy) const { return ( std::fabs( m_xAtYEq0[i_hit] ) < 1.0E-9 ) ? 1. / std::sqrt( 1 + dxDy * dxDy ) : std::cos(dxDy); }
    __host__ __device__ inline bool highThreshold(const int i_hit) const { return m_highThreshold[i_hit]; }
    __host__ __device__ inline bool isYCompatible( const int i_hit, const float y, const float tol ) const { return yMin(i_hit) - tol <= y && y <= yMax(i_hit) + tol; }
    __host__ __device__ inline bool isNotYCompatible( const int i_hit, const float y, const float tol ) const { return yMin(i_hit) - tol > y || y > yMax(i_hit) + tol; }
    __host__ __device__ inline int LHCbID(const int i_hit) const { return m_LHCbID[i_hit]; }
    __host__ __device__ inline float sinT(const int i_hit, const float dxDy) const { return tanT(i_hit) * cosT(i_hit, dxDy); }
    __host__ __device__ inline float tanT(const float dxDy) const { return -1 * dxDy; }
    __host__ __device__ inline float weight(const int i_hit) const { return m_weight[i_hit]; }
    __host__ __device__  inline float xAt( const int i_hit, const float globalY, const float dxDy ) const { return m_xAtYEq0[i_hit] + globalY * dxDy; }
    __host__ __device__ inline float xAtYEq0(const int i_hit) const { return m_xAtYEq0[i_hit]; }
    __host__ __device__ inline float xMax(const int i_hit, const float dxDy) const { return std::max( xAt(i_hit, yBegin(i_hit), dxDy), xAt(i_hit, yEnd(i_hit), dxDy) ); }
    __host__ __device__ inline float xMin(const int i_hit, const float dxDy) const { return std::min( xAt(i_hit, yBegin(i_hit), dxDy), xAt(i_hit, yEnd(i_hit), dxDy) ); }
    __host__ __device__ inline float yBegin(const int i_hit) const { return m_yBegin[i_hit]; }
    __host__ __device__ inline float yEnd(const int i_hit) const { return m_yEnd[i_hit]; }
    __host__ __device__ inline float yMax(const int i_hit) const { return std::max( yBegin(i_hit), yEnd(i_hit) ); }
    __host__ __device__ inline float yMid(const int i_hit) const { return 0.5 * ( yBegin(i_hit) + yEnd(i_hit) ); }
    __host__ __device__ inline float yMin(const int i_hit) const { return std::min( yBegin(i_hit), yEnd(i_hit) ); }
    __host__ __device__ inline float zAtYEq0(const int i_hit) const { return m_zAtYEq0[i_hit]; } 

  };

  struct TrackUT {
    
    unsigned int LHCbIDs[VeloUTTracking::max_track_size];
    float qop;
    unsigned short hitsNum = 0;
    
    
    __host__ __device__ void addLHCbID( unsigned int id ) {
      LHCbIDs[hitsNum++] = id;
    }

    __host__ __device__ void set_qop( float _qop ) {
      qop = _qop;
    }
  };
   
  /* Structure containing indices to hits within hit array */
  struct TrackHits { 
    unsigned short hitsNum = 0;
    unsigned short hits[VeloUTTracking::max_track_size];
    unsigned short veloTrackIndex;
    float qop;

    __host__ __device__ void addHit( const unsigned short _h ) {
      hits[hitsNum] = _h;
      ++hitsNum;
    }
    __host__ __device__ void set_qop( float _qop ) {
      qop = _qop;
    }
  };
  
}

