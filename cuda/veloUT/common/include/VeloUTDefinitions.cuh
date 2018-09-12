#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Common.h"
#include "VeloDefinitions.cuh"
#include "SciFiDefinitions.cuh"

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

   /**
   * @brief Complete state, unlike the reduced VELO one
   *
   *        {x, y, tx, ty, qOverP}
   *
   *        c00 c10 c20 c30 c40
   *            c11 c21 c31 c41
   *                c22 c32 c42
   *                    c33 c43
   *                        c44
   */

 

  /* SoA for hit variables
     The hits for every layer are written behind each other, the offsets 
     are stored for access;
     one Hits structure exists per event
   */
  struct HitsSoA {
    int layer_offset[n_layers];
    int n_hits_layers[n_layers];
    
    float yBegin[max_numhits_per_event];
    float yEnd[max_numhits_per_event];
    float zAtYEq0[max_numhits_per_event];
    float xAtYEq0[max_numhits_per_event];
    float weight[max_numhits_per_event];
    int   highThreshold[max_numhits_per_event];
    unsigned int LHCbID[max_numhits_per_event];
    int planeCode[max_numhits_per_event];

    
    __host__ __device__ inline float cosT(const int i_hit, const float dxDy) const { return ( std::fabs( xAtYEq0[i_hit] ) < 1.0E-9 ) ? 1. / std::sqrt( 1 + dxDy * dxDy ) : std::cos(dxDy); }
    __host__ __device__ inline bool isNotYCompatible( const int i_hit, const float y, const float tol ) const { return yMin(i_hit) - tol > y || y > yMax(i_hit) + tol; }
    __host__ __device__  inline float xAt( const int i_hit, const float globalY, const float dxDy ) const { return xAtYEq0[i_hit] + globalY * dxDy; }
    __host__ __device__ inline float yMax(const int i_hit) const { return std::max( yBegin[i_hit], yEnd[i_hit] ); }
    __host__ __device__ inline float yMin(const int i_hit) const { return std::min( yBegin[i_hit], yEnd[i_hit] ); }

  };

  struct TrackUT {
    
    unsigned int LHCbIDs[VeloUTTracking::max_track_size];
    float qop;
    unsigned short hitsNum = 0;
    unsigned short veloTrackIndex;
    
    __host__ __device__ void addLHCbID( unsigned int id ) {
      LHCbIDs[hitsNum++] = id;
    }

    __host__ __device__ void set_qop( float _qop ) {
      qop = _qop;
    }
  };

}

