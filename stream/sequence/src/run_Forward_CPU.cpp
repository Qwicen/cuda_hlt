#include "../include/run_Forward_CPU.h"

#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"

std::vector< std::vector< VeloUTTracking::TrackVeloUT > > run_forward_on_CPU (
  std::vector< trackChecker::Tracks > * forward_tracks_events,
  ForwardTracking::HitsSoAFwd * hits_layers_events,
  const uint32_t n_hits_layers_events[][ForwardTracking::n_layers],
  std::vector< std::vector< VeloUTTracking::TrackVeloUT > > ut_tracks,
  const int &number_of_events
) {

  PrForward forward;
 
  // Histograms only for checking and debugging
  TFile *f = new TFile("../output/Forward.root", "RECREATE");
  TTree *t_Forward_tracks = new TTree("Forward_tracks", "Forward_tracks");
  float x,z,w,dxdy,yMin,yMax;
  unsigned int LHCbID;
  float x_hit, y_hit, z_hit;
  float first_x, first_y, first_z;
  float last_x, last_y, last_z;
  float qop;
  
  
  t_Forward_tracks->Branch("qop", &qop);

  std::vector<std::vector< VeloUTTracking::TrackVeloUT > > forward_tracks_all;
   
  if ( !forward.initialize() ) {
    error_cout << "Could not initialize Forward" << std::endl;
    return forward_tracks_all;
  }

  for ( int i_event = 0; i_event < number_of_events; ++i_event ) {

    std::vector< VeloUTTracking::TrackVeloUT > forward_tracks = forward(ut_tracks[i_event], &(hits_layers_events[i_event]), n_hits_layers_events[i_event]);
    
    // store qop in tree
    for ( auto Forward_track : forward_tracks ) {
      qop = Forward_track.track.qop;
      t_Forward_tracks->Fill();
    }
    
    // save in format for track checker
    std::vector< VeloUTTracking::TrackUT > forward_tracks_reduced;
    for ( auto veloUT_track : forward_tracks ) {
      forward_tracks_reduced.push_back(veloUT_track.track);
    }
    trackChecker::Tracks checker_tracks = prepareForwardTracks( forward_tracks_reduced );
    //debug_cout << "Passing " << checker_tracks.size() << " tracks to PrChecker" << std::endl;
    
    forward_tracks_events->emplace_back( checker_tracks );
    forward_tracks_all.push_back(forward_tracks);
    
  }
  
  
  f->Write();
  f->Close();
  
  return forward_tracks_all;
}
