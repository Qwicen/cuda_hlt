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

  struct Hits {
    float x[max_numhits_per_layer];
    float z[max_numhits_per_layer];
    float yMin[max_numhits_per_layer];
    float yMax[max_numhits_per_layer];
    float dxdy[max_numhits_per_layer];
    float zAtyEq0[max_numhits_per_layer];
    float weight[max_numhits_per_layer];
    int   planeCode[max_numhits_per_layer];
    int   highThreshold[max_numhits_per_layer];
  };



  





}
