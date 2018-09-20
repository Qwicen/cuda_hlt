#include "run_Forward_CPU.h"

#ifdef WITH_ROOT
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#endif

int run_forward_on_CPU (
  std::vector< trackChecker::Tracks >& forward_tracks_events,
  SciFi::HitsSoA* hits_layers_events,
  uint* host_velo_tracks_atomics,
  uint* host_velo_track_hit_number,
  uint* host_velo_states,
  VeloUTTracking::TrackUT * veloUT_tracks,
  const int * n_veloUT_tracks_events,
  const uint &number_of_events
) {

#ifdef WITH_ROOT
  // Histograms only for checking and debugging
  TFile *f = new TFile("../output/Forward.root", "RECREATE");
  TTree *t_Forward_tracks = new TTree("Forward_tracks", "Forward_tracks");
  TTree *t_statistics = new TTree("statistics", "statistics");
  float x,z,w,dxdy,yMin,yMax;
  unsigned int LHCbID;
  float x_hit, y_hit, z_hit;
  float first_x, first_y, first_z;
  float last_x, last_y, last_z;
  float qop;
  int n_tracks;
    
  t_Forward_tracks->Branch("qop", &qop);
  t_statistics->Branch("n_tracks", &n_tracks);
#endif

  for ( uint i_event = 0; i_event < number_of_events; ++i_event ) {

    // Velo consolidated types
    const Velo::Consolidated::Tracks velo_tracks {(uint*) host_velo_tracks_atomics, host_velo_track_hit_number, i_event, number_of_events};
    const uint event_tracks_offset = velo_tracks.tracks_offset(i_event);
    const Velo::Consolidated::States host_velo_states_event {host_velo_states, velo_tracks.total_number_of_tracks};

    int n_forward_tracks = 0;
    SciFi::Track forward_tracks[SciFi::max_tracks];
    
    // initialize TMVA vars
    SciFi::Tracking::TMVA tmva1;
    SciFi::Tracking::TMVA1_Init( tmva1 );
    SciFi::Tracking::TMVA tmva2;
    SciFi::Tracking::TMVA2_Init( tmva2 );

    SciFi::Tracking::Arrays constArrays;
 
    PrForwardWrapper(
      &(hits_layers_events[i_event]),
      host_velo_states_event,
      event_tracks_offset,
      veloUT_tracks + i_event * VeloUTTracking::max_num_tracks,
      n_veloUT_tracks_events[i_event],
      tmva1,
      tmva2,
      constArrays,
      forward_tracks,
      n_forward_tracks);

       
#ifdef WITH_ROOT
    // store qop in tree
    for ( int i_track = 0; i_track < n_forward_tracks; ++i_track ) {
      qop = forward_tracks[i_track].qop;
      t_Forward_tracks->Fill();
    }
    n_tracks = n_forward_tracks;
    t_statistics->Fill();
#endif
    
    // save in format for track checker
    trackChecker::Tracks checker_tracks = prepareForwardTracks( forward_tracks, n_forward_tracks );
    
    forward_tracks_events.emplace_back( checker_tracks );

  }
  
#ifdef WITH_ROOT
  f->Write();
  f->Close();
#endif
  
  return 0;
}
