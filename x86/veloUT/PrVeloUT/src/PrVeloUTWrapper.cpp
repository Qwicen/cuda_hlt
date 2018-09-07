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
void call_PrVeloUT (
  const uint* velo_track_hit_number,
  const VeloTracking::Hit<mc_check_enabled>* velo_track_hits,
  const int number_of_tracks_event,
  const int accumulated_tracks_event,
  const VeloState* velo_states_event,
  VeloUTTracking::HitsSoA *hits_layers,
  const PrUTMagnetTool *magnet_tool,
  VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks],
  std::vector<VeloUTTracking::TrackVeloUT>& outputTracks,
  int &n_velo_tracks_in_UT,
  int &n_veloUT_tracks )
{
  
  int posLayers[4][85];
  fillIterators(hits_layers, posLayers);

  const float* fudgeFactors = &(magnet_tool->dxLayTable[0]);
  const float* bdlTable     = &(magnet_tool->bdlTable[0]);

  // array to store indices of selected hits in layers
  // -> can then access the hit information in the HitsSoA
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers];
  for ( int i_track = 0; i_track < number_of_tracks_event; ++i_track ) {
    if ( velo_states_event[i_track].backward ) continue;
    
    if( !veloTrackInUTAcceptance( velo_states_event[i_track] ) ) continue;
    n_velo_tracks_in_UT++;

    // for storing calculated x position of hits for this track
    float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];

    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      n_hitCandidatesInLayers[i_layer] = 0;
    }
    if( !getHits(
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          x_pos_layers,
          posLayers,
          hits_layers,
          fudgeFactors,
          velo_states_event[i_track] ) ) continue;
    
    TrackHelper helper(velo_states_event[i_track]);

    // indices within hitCandidatesInLayers for selected hits belonging to best track 
    int hitCandidateIndices[VeloUTTracking::n_layers];
    
    // go through UT layers in forward direction
    if( !formClusters(
          hitCandidatesInLayers,
          n_hitCandidatesInLayers,
          x_pos_layers,
          hitCandidateIndices,
          hits_layers,
          helper,
          true) ){
      
      // go through UT layers in backward direction
      formClusters(
        hitCandidatesInLayers,
        n_hitCandidatesInLayers,
        x_pos_layers,
        hitCandidateIndices,
        hits_layers,
        helper,
        false);
    }

    int n_tracks_prev = n_veloUT_tracks;
    if ( helper.n_hits > 0 ) {
      prepareOutputTrack(
        velo_track_hit_number,
        velo_track_hits,
        accumulated_tracks_event,
        i_track,
        helper,
        hitCandidatesInLayers,
        n_hitCandidatesInLayers,
        hits_layers,
        x_pos_layers,
        hitCandidateIndices,
        VeloUT_tracks,
        &n_veloUT_tracks,
        bdlTable);

      // prepare output tracks needed for forward tracking
      if ( n_veloUT_tracks > n_tracks_prev ) {
        VeloUTTracking::TrackVeloUT outputtrack;
        outputtrack.track = VeloUT_tracks[n_veloUT_tracks - 1];
        outputtrack.state_endvelo.x = helper.state.x;
        outputtrack.state_endvelo.y = helper.state.y;
        outputtrack.state_endvelo.z = helper.state.z;
        outputtrack.state_endvelo.tx = helper.state.tx;
        outputtrack.state_endvelo.ty = helper.state.ty;
        outputtrack.state_endvelo.chi2 = helper.state.chi2;
        outputtrack.state_endvelo.c00 = helper.state.c00;
        outputtrack.state_endvelo.c20 = helper.state.c20;
        outputtrack.state_endvelo.c22 = helper.state.c22;
        outputtrack.state_endvelo.c11 = helper.state.c11;
        outputtrack.state_endvelo.c31 = helper.state.c31;
        outputtrack.state_endvelo.c33 = helper.state.c33;
        outputtrack.state_endvelo.qOverP = outputtrack.track.qop; 
        outputTracks.emplace_back( outputtrack );
      }
      
    }
  }
 
}
 
