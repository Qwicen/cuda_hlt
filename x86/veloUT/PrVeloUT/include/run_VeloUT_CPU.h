#pragma once

// #include "Common.h"
// #include "TrackChecker.h"
// #include "PrVeloUTWrapper.h"
// #include "Tools.h"
// #include "Sorting.cuh"
// #include "VeloUTDefinitions.cuh"

// int run_veloUT_on_CPU (
//   std::vector<trackChecker::Tracks>& ut_tracks_events,
//   uint* ut_hits,
//   uint* ut_hit_count,
//   const PrUTMagnetTool* host_ut_magnet_tool,
//   const float host_ut_dxDy[VeloUTTracking::n_layers],
//   const Velo::State* host_velo_states,
//   const int * host_accumulated_tracks,
//   const uint* host_velo_track_hit_number_pinned,
//   const Velo::Hit* host_velo_track_hits_pinned,
//   const int * host_number_of_tracks_pinned,
//   const int &number_of_events
// );


// template<class T>
// void applyXPermutation(
//   const uint* permutation,
//   const uint hit_start,
//   const uint number_of_hits,
//   T* container
// ) {
//   T interim_container[number_of_hits];
//   for ( int i_hit = 0; i_hit < number_of_hits; ++i_hit ) {
//     interim_container[i_hit] = container[hit_start + i_hit];
//   }
  
//   // Apply permutation across all hits in the layer
//   for (uint permutation_index = 0; permutation_index < number_of_hits; ++permutation_index) {
//     const auto hit_index = permutation[permutation_index];
//     container[hit_start + permutation_index] = interim_container[hit_index];
//   }
// }
