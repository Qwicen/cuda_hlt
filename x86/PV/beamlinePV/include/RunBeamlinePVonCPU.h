#pragma once

#include "Common.h"
#include "TrackBeamLineVertexFinder.h" 
#include "CpuHandler.cuh" 

void run_BeamlinePV_on_CPU(
  uint* kalmanvelo_states,
  uint * velo_atomics,
  uint* velo_track_hit_number,
  PV::Vertex* reconstructed_pvs, 
  int* number_of_pvs, 
  const uint number_of_events 
);

CPU_ALGORITHM(run_BeamlinePV_on_CPU, cpu_beamlinePV_t)  
