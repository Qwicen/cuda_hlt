#include "RunBeamlinePVonCPU.h"

void run_BeamlinePV_on_CPU(
  uint* kalmanvelo_states,
  uint * velo_atomics,
  uint* velo_track_hit_number,
  const uint number_of_events 
) 
{

  std::vector<PV::Vertex> vertices = findPVs(
    kalmanvelo_states,
    (int*)velo_atomics,
    velo_track_hit_number,
    number_of_events);
  
 
}


