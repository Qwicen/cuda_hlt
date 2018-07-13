#pragma once

#include "../../common/include/VeloUTDefinitions.cuh"

__global__ void veloUT(
  VeloUTTracking::HitsSoA* dev_ut_hits
);
