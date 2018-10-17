#include "PrVeloUTWrapper.h"

//-----------------------------------------------------------------------------
// Implementation file for PrVeloUT
//
// 2007-05-08: Mariusz Witek
// 2017-03-01: Christoph Hasse (adapt to future framework)
// 2018-05-05: Plácido Fernández (make standalone)
// 2018-07:    Dorothea vom Bruch (convert to C code for GPU compatability)
//-----------------------------------------------------------------------------

//=============================================================================
// Main execution
//=============================================================================
// void call_PrVeloUT (
//   const uint* velo_track_hit_number,
//   const Velo::Hit* velo_track_hits,
//   const int number_of_tracks_event,
//   const int accumulated_tracks_event,
//   const Velo::State* velo_states_event,
//   UTHits& ut_hits,
//   UTHitCount& ut_hit_count,
//   const PrUTMagnetTool *magnet_tool,
//   const float* ut_dxDy,
//   VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks],
//   int &n_velo_tracks_in_UT,
//   int &n_veloUT_tracks )
// {
//   int posLayers[4][85];
//   fillIterators(ut_hits, ut_hit_count, posLayers);

//   const float* fudgeFactors = &(magnet_tool->dxLayTable[0]);
//   const float* bdlTable     = &(magnet_tool->bdlTable[0]);

//   // array to store indices of selected hits in layers
//   // -> can then access the hit information in the HitsSoA
//   int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
//   int n_hitCandidatesInLayers[VeloUTTracking::n_layers];
//   for ( int i_track = 0; i_track < number_of_tracks_event; ++i_track ) {
//     if ( velo_states_event[i_track].backward ) continue;
    
//     if( !veloTrackInUTAcceptance( velo_states_event[i_track] ) ) continue;
//     n_velo_tracks_in_UT++;

//     // for storing calculated x position of hits for this track
//     float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];

//     for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
//       n_hitCandidatesInLayers[i_layer] = 0;
//     }
//     if( !getHits(
//           hitCandidatesInLayers,
//           n_hitCandidatesInLayers,
//           x_pos_layers,
//           posLayers,
//           ut_hits,
//           ut_hit_count,
//           fudgeFactors,
//           velo_states_event[i_track],
//           ut_dxDy ) ) continue;
    
//     TrackHelper helper(velo_states_event[i_track]);

//     // indices within hitCandidatesInLayers for selected hits belonging to best track 
//     int hitCandidateIndices[VeloUTTracking::n_layers];
    
//     // go through UT layers in forward direction
//     if( !formClusters(
//           hitCandidatesInLayers,
//           n_hitCandidatesInLayers,
//           x_pos_layers,
//           hitCandidateIndices,
//           ut_hits,
//           ut_hit_count,
//           helper,
//           ut_dxDy,
//           true) ){
      
//       // go through UT layers in backward direction
//       formClusters(
//         hitCandidatesInLayers,
//         n_hitCandidatesInLayers,
//         x_pos_layers,
//         hitCandidateIndices,
//         ut_hits,
//         ut_hit_count,
//         helper,
//         ut_dxDy,
//         false);
//     }
    
//     if ( helper.n_hits > 0 ) {
//       prepareOutputTrack(
//         velo_track_hit_number,
//         velo_track_hits,
//         accumulated_tracks_event,
//         i_track,
//         helper,
//         hitCandidatesInLayers,
//         n_hitCandidatesInLayers,
//         ut_hits,
//         ut_hit_count,
//         x_pos_layers,
//         hitCandidateIndices,
//         VeloUT_tracks,
//         &n_veloUT_tracks,
//         bdlTable);
//     }
//   }
// }
