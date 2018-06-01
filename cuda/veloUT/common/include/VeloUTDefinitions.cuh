#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../../../main/include/Common.h"

#include "assert.h"

namespace VeloUTTracking {

  static constexpr uint max_numhits_per_layer = 100;
  /* There are two stations with two layers each */
  static constexpr uint n_layers = 4;
  static constexpr uint n_ut_hit_variables = 8;
  


  struct Hits {
    float * x[max_numhits_per_layer];
    float * z[max_numhits_per_layer];
    float * yMin[max_numhits_per_layer];
    float * yMax[max_numhits_per_layer];
    float * dxdy[max_numhits_per_layer];
    float * zAtyEq0[max_numhits_per_layer];
    int   * planeCode[max_numhits_per_layer];
    int   * highThreshold[max_numhits_per_layer];
  };



  





}
