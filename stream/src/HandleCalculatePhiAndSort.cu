#include "HandleCalculatePhiAndSort.cuh"

void CalculatePhiAndSort::operator()() {
  calculatePhiAndSort<<<num_blocks, num_threads, 0, *stream>>>(
    dev_events,
    dev_event_offsets,
    dev_hit_offsets,
    dev_hit_phi,
    dev_hit_temp,
    dev_hit_permutation
  );
}
