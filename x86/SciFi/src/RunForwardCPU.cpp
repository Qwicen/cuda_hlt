#include "RunForwardCPU.h"

#ifdef WITH_ROOT
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#endif

int run_forward_on_CPU (
  SciFi::TrackHits* host_scifi_tracks,
  int* host_scifi_n_tracks,
  const uint* host_scifi_hits,
  const uint* host_scifi_hit_count,
  const char* host_scifi_geometry,
  const std::array<float, 9>& host_inv_clus_res,
  const uint* host_velo_tracks_atomics,
  const uint* host_velo_track_hit_number,
  const char* host_velo_states,
  const int * host_atomics_ut,
  const uint* host_ut_track_hit_number,
  const float* host_ut_qop,
  const uint* host_ut_track_velo_indices,
  const uint number_of_events
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
  float state_x, state_y, state_z, state_tx, state_ty;

  t_Forward_tracks->Branch("qop", &qop);
  t_Forward_tracks->Branch("state_x", &state_x);
  t_Forward_tracks->Branch("state_y", &state_y);
  t_Forward_tracks->Branch("state_z", &state_z);
  t_Forward_tracks->Branch("state_tx", &state_tx);
  t_Forward_tracks->Branch("state_ty", &state_ty);
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
    const Velo::Consolidated::Tracks velo_tracks {(uint*) host_velo_tracks_atomics, (uint*)host_velo_track_hit_number, i_event, number_of_events};
    const uint event_tracks_offset = velo_tracks.tracks_offset(i_event);
    const Velo::Consolidated::States host_velo_states_event {(char*)host_velo_states, velo_tracks.total_number_of_tracks};

    // UT consolidated types
    UT::Consolidated::Tracks ut_tracks {
     (uint*)host_atomics_ut,
     (uint*)host_ut_track_hit_number,
     (float*)host_ut_qop,
     (uint*)host_ut_track_velo_indices,
      i_event,
      number_of_events
      };
    const int n_veloUT_tracks_event = ut_tracks.number_of_tracks(i_event);
    
    // SciFi non-consolidated types
    int* n_forward_tracks = host_scifi_n_tracks + i_event;
    SciFi::TrackHits* scifi_tracks_event = host_scifi_tracks + i_event * SciFi::Constants::max_tracks;

    const uint total_number_of_hits = host_scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats]; 
    SciFi::HitCount scifi_hit_count {(uint32_t*)host_scifi_hit_count, i_event};
    
    const SciFi::SciFiGeometry scifi_geometry(host_scifi_geometry);
    
    SciFi::Hits scifi_hits(
     (uint*)host_scifi_hits,
     total_number_of_hits,
     &scifi_geometry, 
     reinterpret_cast<const float*>(host_inv_clus_res.data()));

#ifdef WITH_ROOT
    // store hit variables in tree
    for(size_t zone = 0; zone < SciFi::Constants::n_zones; zone++) {
      const auto zone_offset = scifi_hit_count.zone_offset(zone);
      for(size_t hit = 0; hit < scifi_hit_count.zone_number_of_hits(zone); hit++) {
        const auto hit_offset = zone_offset + hit;

        planeCode = scifi_hits.planeCode(hit_offset);
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
      ut_tracks,
      n_veloUT_tracks_event,
      &tmva1,
      &tmva2,
      &constArrays,
      scifi_tracks_event,
      (uint*)n_forward_tracks);

#ifdef WITH_ROOT
    // store qop in tree
    for ( int i_track = 0; i_track < *n_forward_tracks; ++i_track ) {
      qop = scifi_tracks_event[i_track].qop;
      state_x  = scifi_tracks_event[i_track].state.x;
      state_y  = scifi_tracks_event[i_track].state.y;
      state_z  = scifi_tracks_event[i_track].state.z;
      state_tx = scifi_tracks_event[i_track].state.tx;
      state_ty = scifi_tracks_event[i_track].state.ty;
      t_Forward_tracks->Fill();
    }
    n_tracks = n_forward_tracks[i_event];
    t_statistics->Fill();
#endif
  }

#ifdef WITH_ROOT
  f->Write();
  f->Close();
#endif

  return 0;
}
