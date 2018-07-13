#pragma once

#include "../../../main/include/Common.h"

#include "../../../checker/lib/include/TrackChecker.h"

#include "../../../x86/veloUT/PrVeloUT/include/PrVeloUT.h"
#include "../../../main/include/Tools.h"

int run_veloUT_on_CPU (
  std::vector< trackChecker::Tracks > * ut_tracks_events,
  VeloUTTracking::HitsSoA * hits_layers_events,
  const uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
  const VeloState * host_velo_states,
  const int * host_accumulated_tracks,
  const uint* host_velo_track_hit_number_pinned,
  const VeloTracking::Hit<true>* host_velo_track_hits_pinned,
  const int * host_number_of_tracks_pinned,
  const int &number_of_events
);

void findPermutation(
  const float* hit_Xs,
  const uint hit_start,
  uint* hit_permutations,
  const uint n_hits
); 

template<class T>
void applyXPermutation(
  const uint* permutation,
  const uint hit_start,
  const uint number_of_hits,
  T* container
) {
  // To do: find better solution such that not all arrays have to be copied
  T interim_container[number_of_hits];
  for ( int i_hit = 0; i_hit < number_of_hits; ++i_hit ) {
    interim_container[i_hit] = container[hit_start + i_hit];
  }
  
  // Apply permutation across all hits in the layer
  for (uint permutation_index = 0; permutation_index < number_of_hits; ++permutation_index) {
    const auto hit_index = permutation[permutation_index];
    container[hit_start + permutation_index] = interim_container[hit_index];
  }
  
}
