#include "HandleConsolidateTracks.cuh"

void ConsolidateTracks::operator()() {
  consolidate_tracks<<<num_blocks, num_threads, 0, *stream>>>(
    dev_atomics_storage,
    dev_tracks,
    dev_output_tracks
  );
}
