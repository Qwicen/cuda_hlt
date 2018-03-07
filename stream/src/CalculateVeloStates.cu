#include "../include/CalculateVeloStates.cuh"

void CalculateVeloStates::operator()() {
  velo_fit<<<num_blocks, num_threads, 0, stream>>>(
    dev_events,
    dev_atomics_storage,
    dev_tracks,
    dev_velo_states,
    dev_hit_temp,
    dev_event_offsets,
    dev_hit_offsets
  );
}
