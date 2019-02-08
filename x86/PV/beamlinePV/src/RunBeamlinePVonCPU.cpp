#include "RunBeamlinePVOnCPU.h"

void run_BeamlinePV_on_CPU(
  char* kalmanvelo_states,
  uint* velo_atomics,
  uint* velo_track_hit_number,
  PV::Vertex* reconstructed_pvs,
  int* number_of_pvs,
  const uint number_of_events)
{

  findPVs(
    kalmanvelo_states, (int*) velo_atomics, velo_track_hit_number, reconstructed_pvs, number_of_pvs, number_of_events);
}
