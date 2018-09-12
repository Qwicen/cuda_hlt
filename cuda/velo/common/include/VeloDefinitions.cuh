#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Common.h"
#include "assert.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <stdint.h>
#include <stdio.h>

namespace VeloTracking {

// Detector constants
static constexpr uint n_modules = 52;
static constexpr uint n_sensors = n_modules * 4;
static constexpr float z_endVelo = 770;

// How many concurrent h1s to process max
// It should be a divisor of NUMTHREADS_X
static constexpr uint max_concurrent_h1 = 16;

// Number of concurrent h1s in the first iteration
// The first iteration has no flagged hits and more triplets per hit
static constexpr uint max_concurrent_h1_first_iteration = 8;

// These parameters impact the found tracks
// Maximum / minimum acceptable phi
// This impacts enourmously the speed of track seeding
static constexpr float phi_extrapolation = 0.062f;

// Tolerance angle for forming triplets
static constexpr float max_slope = 0.4f;
static constexpr float tolerance = 0.6f;

// Maximum scatter of each three hits
// This impacts velo tracks and a to a lesser extent
// long and long strange tracks
static constexpr float max_scatter_seeding = 0.004f;

// Making a bigger forwarding scatter window causes
// less clones and more ghosts
static constexpr float max_scatter_forwarding = 0.004f;
// Maximum number of skipped modules allowed for a track
// before storing it
static constexpr uint max_skipped_modules = 1;

// Total number of atomics required
// This is just a constant
static constexpr uint num_atomics = 5;

// Constants for requested storage on device
static constexpr uint max_tracks = 1200;
static constexpr uint max_track_size = 26;
static constexpr uint max_numhits_in_module = 350;

// Maximum number of tracks to follow at a time
static constexpr uint ttf_modulo = 2000;
static constexpr uint max_weak_tracks = 500;

// High number of hits per event
static constexpr uint max_number_of_hits_per_event = 9500;

// Constants for filters
static constexpr uint states_per_track = 3; 
static constexpr float param_w = 3966.94f;
static constexpr float param_w_inverted = 0.000252083f;

// Max chi2
static constexpr float max_chi2 = 20.0;

struct Module {
  uint hitStart;
  uint hitNums;

  __device__ Module(){}
  __device__ Module(
    const uint _hitStart,
    const uint _hitNums
  ) : hitStart(_hitStart), hitNums(_hitNums) {}
};

struct HitBase { // 3 * 4 = 12 B
  float x;
  float y;
  float z;
    
  __device__ HitBase(){}
  __device__ HitBase(
    const float _x,
    const float _y,
    const float _z
  ) : x(_x), y(_y), z(_z) {}
};

template <bool MCCheck>
struct Hit;

template <>
struct Hit <true> : public HitBase { // 4 * 4 = 16 B
  uint32_t LHCbID;
  
  __device__ Hit(){}
  __device__ Hit(
    const float _x,
    const float _y,
    const float _z,
    const uint32_t _LHCbID
  ) : HitBase( _x, _y, _z ), LHCbID( _LHCbID ) {}
};

template <>
struct Hit <false> : public HitBase { // 4 * 3 = 12 B
   __device__ Hit(){}
   __device__ Hit(
     const float _x,
     const float _y,
     const float _z
  ) : HitBase( _x, _y, _z) {}
};

/**
 * @brief TrackletHits struct
 */
struct TrackletHits {
  unsigned short hits[3];

  __device__ TrackletHits(){}
  __device__ TrackletHits(
    const unsigned short h0,
    const unsigned short h1,
    const unsigned short h2
  ) {
    hits[0] = h0;
    hits[1] = h1;
    hits[2] = h2;
  }
};

/* Structure containing indices to hits within hit array */
struct TrackHits { // 2 + 26 * 2 = 54 B
  unsigned short hitsNum;
  unsigned short hits[VeloTracking::max_track_size];

  __device__ TrackHits(){}
  
  __device__ TrackHits(
    const unsigned short _hitsNum,
    const unsigned short _h0,
    const unsigned short _h1,
    const unsigned short _h2
  ) : hitsNum(_hitsNum) {
    hits[0] = _h0;
    hits[1] = _h1;
    hits[2] = _h2;
  }

  __device__ TrackHits(const TrackletHits& tracklet) {
    hitsNum = 3;
    hits[0] = tracklet.hits[0];
    hits[1] = tracklet.hits[1];
    hits[2] = tracklet.hits[2];
  }
};

/* Structure to save final track
   Contains information needed later on in the HLT chain
   and / or for truth matching

   Without MC: 4 + 26 * 12 = 316 B
   With MC:    4 + 26 * 16 = 420 B */
template<bool MCCheck>   
struct Track {
  bool backward;
  unsigned short hitsNum;
  Hit<MCCheck> hits[VeloTracking::max_track_size];
  
    __device__ Track(){
      hitsNum = 0;
    }
    
    __device__ void addHit( Hit <MCCheck> _h ){
      hits[ hitsNum ] = _h;
      hitsNum++;
    }
  };
} // VeloTracking namespace

/**
 * @brief A simplified state for the Velo
 *        
 *        {x, y, tx, ty, 0}
 *        
 *        associated with a simplified covariance
 *        since we do two fits (one in X, one in Y)
 *
 *        c00 0.f c20 0.f 0.f
 *            c11 0.f c31 0.f
 *                c22 0.f 0.f
 *                    c33 0.f
 *                        0.f
 */

struct VeloState { // 48 B
  float x, y, tx, ty;
  float c00, c20, c22, c11, c31, c33;
  float chi2;
  float z;
  bool backward;
};

/**
 * @brief Means square fit parameters
 *        required for Kalman fit (Velo)
 */
struct TrackFitParameters {
  float tx, ty;
  bool backward;
};
