#include "run_Forward_CPU.h"

#ifdef WITH_ROOT
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#endif

int run_forward_on_CPU (
  std::vector< trackChecker::Tracks >& forward_tracks_events,
  uint* host_scifi_hits,
  uint* host_scifi_hit_count,
  uint* host_velo_tracks_atomics,
  uint* host_velo_track_hit_number,
  uint* host_velo_states,
  VeloUTTracking::TrackUT * veloUT_tracks,
  const int * n_veloUT_tracks_events,
  const uint &number_of_events
) {

#ifdef WITH_ROOT
  // Histograms only for checking and debugging
  TFile *f = new TFile("../output/scifi.root", "RECREATE");
  TTree *t_Forward_tracks = new TTree("Forward_tracks", "Forward_tracks");
  TTree *t_statistics = new TTree("statistics", "statistics");
  TTree *t_scifi_hits = new TTree("scifi_hits","scifi_hits");
  uint planeCode, hitZone, LHCbID;
  float x0, z0, w, dxdy, dzdy, yMin, yMax;
  float qop;
  int n_tracks;
    
  t_Forward_tracks->Branch("qop", &qop);
  t_statistics->Branch("n_tracks", &n_tracks);
  t_scifi_hits->Branch("planeCode", &planeCode);
  t_scifi_hits->Branch("hitZone", &hitZone);
  t_scifi_hits->Branch("LHCbID", &LHCbID);
  t_scifi_hits->Branch("x0", &x0);
  t_scifi_hits->Branch("z0", &z0);
  t_scifi_hits->Branch("w", &w);
  t_scifi_hits->Branch("dxdy", &dxdy);
  t_scifi_hits->Branch("dzdy", &dzdy);
  t_scifi_hits->Branch("yMin", &yMin);
  t_scifi_hits->Branch("yMax", &yMax);
#endif

  for ( uint i_event = 0; i_event < number_of_events; ++i_event ) {

    // Velo consolidated types
    const Velo::Consolidated::Tracks velo_tracks {(uint*) host_velo_tracks_atomics, host_velo_track_hit_number, i_event, number_of_events};
    const uint event_tracks_offset = velo_tracks.tracks_offset(i_event);
    const Velo::Consolidated::States host_velo_states_event {host_velo_states, velo_tracks.total_number_of_tracks};

    uint n_forward_tracks = 0;
    SciFi::Track forward_tracks[SciFi::max_tracks];

    SciFi::SciFiHitCount scifi_hit_count;
    scifi_hit_count.typecast_after_prefix_sum(host_scifi_hit_count, i_event, number_of_events);

    const uint total_number_of_hits = host_scifi_hit_count[number_of_events * SciFi::Constants::n_zones];
    SciFi::SciFiHits scifi_hits; 
    scifi_hits.typecast_sorted((uint*) host_scifi_hits, total_number_of_hits);

#ifdef WITH_ROOT
    // store hit variables in tree
    for(size_t zone = 0; zone < SciFi::Constants::n_zones; zone++) {
      const auto zone_offset = scifi_hit_count.layer_offsets[zone];
      for(size_t hit = 0; hit < scifi_hit_count.n_hits_layers[zone]; hit++) {
        auto h = scifi_hits.getHit(zone_offset + hit);
        planeCode = h.planeCode;
        hitZone = h.hitZone;
        LHCbID = h.LHCbID;
        x0 = h.x0;
        z0 = h.z0;
        w  = h.w;
        dxdy = h.dxdy;
        dzdy = h.dzdy;
        yMin = h.yMin;
        yMax = h.yMax;
        t_scifi_hits->Fill();
      }
    }
#endif
    
    
    // initialize TMVA vars
    SciFi::Tracking::TMVA tmva1;
    SciFi::Tracking::TMVA1_Init( tmva1 );
    SciFi::Tracking::TMVA tmva2;
    SciFi::Tracking::TMVA2_Init( tmva2 );

    SciFi::Tracking::Arrays constArrays;
 
    PrForwardWrapper(
      scifi_hits,
      scifi_hit_count,
      host_velo_states_event,
      event_tracks_offset,
      veloUT_tracks + i_event * VeloUTTracking::max_num_tracks,
      n_veloUT_tracks_events[i_event],
      &tmva1,
      &tmva2,
      &constArrays,
      forward_tracks,
      &n_forward_tracks);

       
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
    trackChecker::Tracks checker_tracks = prepareForwardTracksEvent( forward_tracks, n_forward_tracks );
    
    forward_tracks_events.emplace_back( checker_tracks );

  }
  
#ifdef WITH_ROOT
  f->Write();
  f->Close();
#endif
  
  return 0;
}
