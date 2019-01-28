#pragma once

#include "Common.h"
#include "TrackBeamLineVertexFinder.cuh"
#include "CpuHandler.cuh" 

void run_BeamlinePV_on_CPU(
  char* kalmanvelo_states,
  uint * velo_atomics,
  uint* velo_track_hit_number,
  PV::Vertex* reconstructed_pvs, 
  int* number_of_pvs, 
  const uint number_of_events 
);

CPU_ALGORITHM(run_BeamlinePV_on_CPU, cpu_pv_beamline_t)
