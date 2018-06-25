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
    float cos[max_numhits_per_event];
    float yBegin[max_numhits_per_event];
    float yEnd[max_numhits_per_event];
    float dxDy[max_numhits_per_event];
    float zAtYEq0[max_numhits_per_event];
    float xAtYEq0[max_numhits_per_event];
    float weight[max_numhits_per_event];
    int   highThreshold[max_numhits_per_event];
    unsigned int LHCbID[max_numhits_per_event];
  };



  





}
