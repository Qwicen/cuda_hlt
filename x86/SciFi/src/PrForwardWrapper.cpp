#include "PrForwardWrapper.h"

void PrForwardWrapper(
  const SciFi::SciFiHits& hits_layers,
  const Velo::Consolidated::States& velo_states,
  const uint event_tracks_offset,
  const VeloUTTracking::TrackUT * veloUT_tracks,
  const int n_veloUT_tracks,
  SciFi::Tracking::TMVA* tmva1,
  SciFi::Tracking::TMVA* tmva2,
  SciFi::Tracking::Arrays* constArrays,
  SciFi::Track outputTracks[SciFi::max_tracks],
  uint* n_forward_tracks)
{
  
  // Loop over the veloUT input tracks
  for ( int i_veloUT_track = 0; i_veloUT_track < n_veloUT_tracks; ++i_veloUT_track ) {
    const VeloUTTracking::TrackUT& veloUTTr = veloUT_tracks[i_veloUT_track];

    const uint velo_states_index = event_tracks_offset + veloUTTr.veloTrackIndex;
    const MiniState velo_state {velo_states, velo_states_index};
    
    // TODO: hits_layers should point to the event

    find_forward_tracks(
      hits_layers,
      veloUTTr,
      outputTracks,
      n_forward_tracks,
      tmva1,
      tmva2,
      constArrays,
      velo_state);
    
  }

}
