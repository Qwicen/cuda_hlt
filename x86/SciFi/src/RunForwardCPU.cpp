#include "RunForwardCPU.h"

#ifdef WITH_ROOT
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#endif

int run_forward_on_CPU (
  std::vector< trackChecker::Tracks >& forward_tracks_events,
  uint* host_scifi_hits,
  uint* host_scifi_hit_count,
  const char* host_scifi_geometry,
  uint* host_velo_tracks_atomics,
  uint* host_velo_track_hit_number,
  uint* host_velo_states,
  VeloUTTracking::TrackUT * veloUT_tracks,
  const int * n_veloUT_tracks_events,
  const uint number_of_events,
  const std::array<float, 9>& host_inv_clus_res
) {

#ifdef WITH_ROOT
  // Histograms only for checking and debugging
  TFile *f = new TFile("../output/scifi.root", "RECREATE");
  TTree *t_Forward_tracks = new TTree("Forward_tracks", "Forward_tracks");
  TTree *t_statistics = new TTree("statistics", "statistics");
  TTree *t_scifi_hits = new TTree("scifi_hits","scifi_hits");
  uint planeCode, LHCbID;
  float x0, z0, w, dxdy, dzdy, yMin, yMax;
  float qop;
  int n_tracks;

  t_Forward_tracks->Branch("qop", &qop);
  t_statistics->Branch("n_tracks", &n_tracks);
  t_scifi_hits->Branch("planeCode", &planeCode);
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

    const uint total_number_of_hits = host_scifi_hit_count[number_of_events * SciFi::Constants::n_mats];
    const SciFi::SciFiGeometry scifi_geometry(host_scifi_geometry);
    SciFi::SciFiHits scifi_hits(host_scifi_hits,
      total_number_of_hits,
      &scifi_geometry, 
      reinterpret_cast<const float*>(host_inv_clus_res.data()));

#ifdef WITH_ROOT
    // store hit variables in tree
    for(size_t mat = 0; mat < SciFi::Constants::n_mats; mat++) {
      const auto zone_offset = scifi_hit_count.mat_offsets[mat];
      for(size_t hit = 0; hit < scifi_hit_count.n_hits_mats[mat]; hit++) {
        const auto hit_offset = zone_offset + hit;

        planeCode = scifi_hits.planeCode(hit_offset);
        // hitZone = scifi_hits.planeCode(hit_offset);
        LHCbID = scifi_hits.LHCbID(hit_offset);
        x0 = scifi_hits.x0[hit_offset];
        z0 = scifi_hits.z0[hit_offset];
        w  = scifi_hits.w(hit_offset);
        dxdy = scifi_hits.dxdy(hit_offset);
        dzdy = scifi_hits.dzdy(hit_offset);
        yMin = scifi_hits.yMin(hit_offset);
        yMax = scifi_hits.yMax(hit_offset);
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
    trackChecker::Tracks checker_tracks = prepareTracksSingleEvent<TrackCheckerForward, SciFi::Track>( forward_tracks, n_forward_tracks );
    forward_tracks_events.emplace_back( checker_tracks );
  }

#ifdef WITH_ROOT
  f->Write();
  f->Close();
#endif

  return 0;
}
