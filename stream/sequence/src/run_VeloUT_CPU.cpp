#include "../include/run_VeloUT_CPU.h"

#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
 
int run_veloUT_on_CPU (
  std::vector< trackChecker::Tracks > * ut_tracks_events,
  const VeloUTTracking::HitsSoA * hits_layers_events,
  const uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
  VeloState * host_velo_states,
  int * host_accumulated_tracks,
  VeloTracking::Track <mc_check_enabled> *host_tracks_pinned,
  int * host_number_of_tracks_pinned,
  const int &number_of_events
) {

  PrVeloUT velout;

  
      
  // Histograms only for checking and debugging
  TFile *f = new TFile("../output/veloUT.root", "RECREATE");
  TTree *t_ut_hits = new TTree("ut_hits","ut_hits");
  TTree *t_velo_states = new TTree("velo_states", "velo_states");
  TTree *t_track_hits = new TTree("track_hits", "track_hits");
  TTree *t_veloUT_tracks = new TTree("veloUT_tracks", "veloUT_tracks");
  float cos, yBegin, yEnd, dxDy, zAtYEq0, xAtYEq0, weight;
  float x, y, tx, ty, chi2, z, drdz;
  unsigned int LHCbID;
  int highThreshold, layer;
  int backward;
  float x_hit, y_hit, z_hit;
  float first_x, first_y, first_z;
  float last_x, last_y, last_z;
  float qop;
  
  
  t_ut_hits->Branch("cos", &cos);
  t_ut_hits->Branch("yBegin", &yBegin);
  t_ut_hits->Branch("yEnd", &yEnd);
  t_ut_hits->Branch("dxDy", &dxDy);
  t_ut_hits->Branch("zAtYEq0", &zAtYEq0);
  t_ut_hits->Branch("xAtYEq0", &xAtYEq0);
  t_ut_hits->Branch("weight", &weight);
  t_ut_hits->Branch("LHCbID", &LHCbID);
  t_ut_hits->Branch("highThreshold", &highThreshold);
  t_ut_hits->Branch("layer", &layer);
  t_velo_states->Branch("x", &x);
  t_velo_states->Branch("y", &y);
  t_velo_states->Branch("tx", &tx);
  t_velo_states->Branch("ty", &ty);
  t_velo_states->Branch("chi2", &chi2);
  t_velo_states->Branch("z", &z);
  t_velo_states->Branch("backward", &backward);
  t_velo_states->Branch("drdz", &drdz);
  t_track_hits->Branch("x", &x_hit);
  t_track_hits->Branch("y", &y_hit);
  t_track_hits->Branch("z", &z_hit);
  t_velo_states->Branch("first_x", &first_x);
  t_velo_states->Branch("first_y", &first_y);
  t_velo_states->Branch("first_z", &first_z); 
  t_velo_states->Branch("last_x", &last_x);
  t_velo_states->Branch("last_y", &last_y);
  t_velo_states->Branch("last_z", &last_z); 
  t_veloUT_tracks->Branch("qop", &qop);

   
  if ( !velout.initialize() ) {
    error_cout << "Could not initialize VeloUT" << std::endl;
    return -1;
  }
  
  for ( int i_event = 0; i_event < number_of_events; ++i_event ) {
    // find out offsets for every layer
    int accumulated_hits = 0;
    int accumulated_hits_layers[4];
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      accumulated_hits_layers[i_layer] = accumulated_hits;
      accumulated_hits += n_hits_layers_events[i_event][i_layer];
    }
    // Prepare hits
    std::array<std::vector<VeloUTTracking::Hit>,VeloUTTracking::n_layers> inputHits;
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      inputHits[i_layer].clear();
      int layer_offset = accumulated_hits_layers[i_layer];
      for ( int i_hit = 0; i_hit < n_hits_layers_events[i_event][i_layer]; ++i_hit ) {
	VeloUTTracking::Hit hit;
	hit.m_cos = hits_layers_events[i_event].cos[layer_offset + i_hit];
	hit.m_dxDy = hits_layers_events[i_event].dxDy[layer_offset + i_hit];
	hit.m_weight = hits_layers_events[i_event].weight[layer_offset + i_hit];
	hit.m_xAtYEq0 = hits_layers_events[i_event].xAtYEq0[layer_offset + i_hit];
	hit.m_yBegin = hits_layers_events[i_event].yBegin[layer_offset + i_hit];
	hit.m_yEnd = hits_layers_events[i_event].yEnd[layer_offset + i_hit];
	hit.m_zAtYEq0 = hits_layers_events[i_event].zAtYEq0[layer_offset + i_hit];
	hit.m_LHCbID = hits_layers_events[i_event].LHCbID[layer_offset + i_hit];
	hit.m_planeCode = i_layer;
	hit.m_cluster_threshold = hits_layers_events[i_event].highThreshold[layer_offset + i_hit];
	
	inputHits[i_layer].push_back( hit );
	
	// For tree filling
	cos = hit.m_cos;
	yBegin = hit.m_yBegin;
	yEnd = hit.m_yEnd;
	dxDy = hit.m_dxDy;
	zAtYEq0 = hit.m_zAtYEq0;
	xAtYEq0 = hit.m_xAtYEq0;
	weight = hit.m_weight;
	LHCbID = hit.m_LHCbID;
	highThreshold = hit.m_cluster_threshold;
	layer = i_layer;
	
	t_ut_hits->Fill();
      }
      // sort hits according to xAtYEq0
      std::sort( inputHits[i_layer].begin(), inputHits[i_layer].end(), [](VeloUTTracking::Hit a, VeloUTTracking::Hit b) { return a.xAtYEq0() < b.xAtYEq0(); } );
    }
    
    // Prepare Velo tracks
    VeloState* velo_states_event = host_velo_states + host_accumulated_tracks[i_event];
    VeloTracking::Track<true>* tracks_event = host_tracks_pinned + host_accumulated_tracks[i_event];
    std::vector<VeloUTTracking::TrackVelo> tracks;
    for ( uint i_track = 0; i_track < host_number_of_tracks_pinned[i_event]; i_track++ ) {
      
      VeloUTTracking::TrackVelo track;
      
      VeloUTTracking::TrackUT ut_track;
      const VeloTracking::Track<true> velo_track = tracks_event[i_track];
      backward = (int)velo_track.backward;
      ut_track.hitsNum = velo_track.hitsNum;
      for ( int i_hit = 0; i_hit < velo_track.hitsNum; ++i_hit ) {
	ut_track.LHCbIDs.push_back( velo_track.hits[i_hit].LHCbID );
      }
      track.track = ut_track;
      
      track.state = ( velo_states_event[i_track] );
      
      //////////////////////
      // For tree filling
      //////////////////////
      x = track.state.x;
      y = track.state.y;
      tx = track.state.tx;
      ty = track.state.ty;
      chi2 = track.state.chi2;
      z = track.state.z;
      // study (sign of) (dr/dz) -> track moving away from beamline?
      // drop 1/sqrt(x^2+y^2) to avoid sqrt calculation, no effect on sign
      float dx = velo_track.hits[velo_track.hitsNum - 1].x - velo_track.hits[0].x;
      float dy = velo_track.hits[velo_track.hitsNum - 1].y - velo_track.hits[0].y;
      float dz = velo_track.hits[velo_track.hitsNum - 1].z - velo_track.hits[0].z;
      drdz = velo_track.hits[0].x * dx/dz + velo_track.hits[0].y * dy/dz;
      
      first_x = velo_track.hits[0].x;
      first_y = velo_track.hits[0].y;
      first_z = velo_track.hits[0].z;
      last_x = velo_track.hits[velo_track.hitsNum-1].x;
      last_y = velo_track.hits[velo_track.hitsNum-1].y;
      last_z = velo_track.hits[velo_track.hitsNum-1].z;
      
      t_velo_states->Fill();
      
      /* Get hits on track */
      for ( int i_hit = 0; i_hit < velo_track.hitsNum; ++i_hit ) {
	x_hit = velo_track.hits[i_hit].x;
	y_hit = velo_track.hits[i_hit].y;
	z_hit = velo_track.hits[i_hit].z;
	
	t_track_hits->Fill();
      }
      
      
      if ( velo_track.backward ) continue;
      tracks.push_back( track );
    }
    //debug_cout << "at event " << i_event << ", pass " << tracks.size() << " tracks and " << inputHits[0].size() << " hits in layer 0, " << inputHits[1].size() << " hits in layer 1, " << inputHits[2].size() << " hits in layer 2, " << inputHits[3].size() << " in layer 3 to velout" << std::endl;
    
    std::vector< VeloUTTracking::TrackUT > ut_tracks = velout(tracks, inputHits);
    //debug_cout << "\t got " << (uint)ut_tracks.size() << " tracks from VeloUT " << std::endl;
    
    // store qop in tree
    for ( auto veloUT_track : ut_tracks ) {
      qop = veloUT_track.qop;
      t_veloUT_tracks->Fill();
    }
    
    // save in format for track checker
    trackChecker::Tracks checker_tracks = prepareVeloUTTracks( ut_tracks );
    //debug_cout << "Passing " << checker_tracks.size() << " tracks to PrChecker" << std::endl;
    
    ut_tracks_events->emplace_back( checker_tracks );
    
  }
  
  
  f->Write();
  f->Close();
  
  return 0;
}
