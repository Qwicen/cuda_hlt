#include "run_Forward_CPU.h"

#ifdef WITH_ROOT
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#endif

int run_forward_on_CPU (
  std::vector< trackChecker::Tracks >& forward_tracks_events,
  ForwardTracking::HitsSoAFwd * hits_layers_events,
  std::vector< std::vector< VeloUTTracking::TrackVeloUT > > ut_tracks,
  const int &number_of_events
) {

  PrForward forward;

#ifdef WITH_ROOT
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
#endif

  for ( int i_event = 0; i_event < number_of_events; ++i_event ) {

    std::vector< ForwardTracking::TrackForward > forward_tracks = forward(ut_tracks[i_event], &(hits_layers_events[i_event]));

#ifdef WITH_ROOT
    // store qop in tree
    for ( auto track : forward_tracks ) {
      qop = track.qop;
      t_Forward_tracks->Fill();
    }
#endif
    
    // save in format for track checker
    // CAUTION: only checking Velo and UT LHCbIDs of this track!!
    //std::vector< VeloUTTracking::TrackUT > forward_tracks_reduced;
    //debug_cout << "Making reduced forward tracks out of " << int(forward_tracks.size()) << "forward tracks " << std::endl;
    int i_track = 0;
    // for ( auto veloUT_track : forward_tracks ) {
    //   forward_tracks_reduced.push_back(veloUT_track.track);
    //   //debug_cout << "at track " << std::dec << i_track << std::endl;
    //   for ( int i_hit = 0; i_hit < veloUT_track.track.hitsNum; ++i_hit ) {
    //     //debug_cout<<"\t LHCbIDsForward["<<i_hit<<"] = " << std::hex << veloUT_track.track.LHCbIDs[i_hit]<< std::endl;
    //   }
    //   ++i_track;
    // }
    //debug_cout<<"run_Forward_CPU"<<std::endl;
    //trackChecker::Tracks checker_tracks = prepareForwardTracksVeloUTOnly( forward_tracks_reduced );
    //    debug_cout << "Passing " << std::dec << checker_tracks.size() << " tracks to PrChecker" << std::endl;
    //debug_cout << "# of forward tracks = " << int( forward_tracks.size() ) << ", # of checker tracks = " << int( checker_tracks.size() ) << std::endl;

    trackChecker::Tracks checker_tracks = prepareForwardTracks( forward_tracks );
    
    forward_tracks_events.emplace_back( checker_tracks );

    //debug_cout << "End event loop run_forward_CPU " <<std::endl; 
    
  }
  
#ifdef WITH_ROOT
  f->Write();
  f->Close();
#endif
  
  return 0;
}
