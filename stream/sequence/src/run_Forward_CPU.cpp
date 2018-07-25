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
  TTree *t_forward_hits = new TTree("forward_hits","forward_hits");
  TTree *t_Forward_tracks = new TTree("Forward_tracks", "Forward_tracks");
  float x,z,w,dxdy,yMin,yMax;
  unsigned int LHCbID;
  float x_hit, y_hit, z_hit;
  float first_x, first_y, first_z;
  float last_x, last_y, last_z;
  float qop;
  
  
  t_forward_hits->Branch("x",&x);
  t_Forward_tracks->Branch("qop", &qop);

  std::vector<std::vector< VeloUTTracking::TrackVeloUT > > forward_tracks_all;
   
  if ( !forward.initialize() ) {
    error_cout << "Could not initialize Forward" << std::endl;
    return forward_tracks_all;
  }

  for ( int i_event = 0; i_event < number_of_events; ++i_event ) {

    // Prepare hits
    std::array<std::vector<ForwardTracking::Hit>,ForwardTracking::n_layers> inputHits;
    for ( int i_layer = 0; i_layer < ForwardTracking::n_layers; ++i_layer ) {
      inputHits[i_layer].clear();
      int layer_offset = hits_layers_events[i_event].layer_offset[i_layer];
      uint n_hits = n_hits_layers_events[i_event][i_layer];
      
      for ( int i_hit = 0; i_hit < n_hits; ++i_hit ) {
	ForwardTracking::Hit hit;
	hit.m_x = hits_layers_events[i_event].m_x[layer_offset + i_hit];
	hit.m_z = hits_layers_events[i_event].m_z[layer_offset + i_hit];
	hit.m_w = hits_layers_events[i_event].m_w[layer_offset + i_hit];
	hit.m_dxdy = hits_layers_events[i_event].m_dxdy[layer_offset + i_hit];
	hit.m_yMin = hits_layers_events[i_event].m_yMin[layer_offset + i_hit];
	hit.m_yMax = hits_layers_events[i_event].m_yMax[layer_offset + i_hit];
	hit.m_LHCbID = hits_layers_events[i_event].m_LHCbID[layer_offset + i_hit];
	hit.m_planeCode = hits_layers_events[i_event].m_planeCode[layer_offset + i_hit];
        hit.m_hitZone = hits_layers_events[i_event].m_hitZone[layer_offset + i_hit];
	
	inputHits[i_layer].push_back( hit );
	
	// For tree filling
	x = hit.m_x;
        
	t_forward_hits->Fill();
      }
    }
    
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
