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
static constexpr uint n_regions_in_layer = 3;

/* For now, the planeCode is an attribute of every hit,
   -> check which information
   needs to be saved for the forward tracking
*/
static constexpr int planeCode[n_layers] = {0, 1, 2, 3};
  
/* Cut-offs */
static constexpr uint max_numhits_per_layer = 500;
static constexpr uint max_numhits_per_event = 6000;
static constexpr uint max_hit_candidates_per_layer = 100;
static constexpr uint max_num_tracks = 400; // to do: what is the best / safest value here?
static constexpr uint max_track_size = VeloTracking::max_track_size + 8; // to do: double check what the max # of hits added in UT really is

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

