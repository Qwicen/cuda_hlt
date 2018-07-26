#include "../include/run_VeloUT_CPU.h"

#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"

void findPermutation(
  const float* hit_Xs,
  const uint hit_start,
  uint* hit_permutations,
  const uint n_hits
) {

  for (unsigned int i = 0; i < n_hits; ++i) {
    const int hit_index = hit_start + i;
    const float x = hit_Xs[hit_index];
    
    // Find out local position
    unsigned int position = 0;
    for (unsigned int j = 0; j < n_hits; ++j) {
      const int other_hit_index = hit_start + j;
      const float other_x = hit_Xs[other_hit_index];
      // Stable sorting
      position += x > other_x || ( x == other_x && i > j );
    }
    assert(position < n_hits);
    
    // Store it in hit_permutations 
    hit_permutations[position] = i;
  }
  
}


std::vector< std::vector< VeloUTTracking::TrackVeloUT > > run_veloUT_on_CPU (
  std::vector< trackChecker::Tracks > * ut_tracks_events,
  VeloUTTracking::HitsSoA * hits_layers_events,
  const uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
  VeloState * host_velo_states,
  int * host_accumulated_tracks,
  uint* host_velo_track_hit_number_pinned,
  VeloTracking::Hit<true>* host_velo_track_hits_pinned,   
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
  float cos, yBegin, yEnd, dxDy, zAtYEq0, xAtYEq0, weight2;
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
  t_ut_hits->Branch("weight2", &weight2);
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

  std::vector<std::vector< VeloUTTracking::TrackVeloUT > > ut_tracks_all;  
 
  if ( !velout.initialize() ) {
    error_cout << "Could not initialize VeloUT" << std::endl;
    return ut_tracks_all;
  }
 
  for ( int i_event = 0; i_event < number_of_events; ++i_event ) {

    // Prepare hits
    std::array<std::vector<VeloUTTracking::Hit>,VeloUTTracking::n_layers> inputHits;
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      inputHits[i_layer].clear();
      int layer_offset = hits_layers_events[i_event].layer_offset[i_layer];
      uint n_hits = n_hits_layers_events[i_event][i_layer];
      
      // sort according to xAtyEq0
      uint hit_permutations[ n_hits_layers_events[i_event][i_layer] ];
      findPermutation( 
        hits_layers_events[i_event].m_xAtYEq0,
      	layer_offset,
      	hit_permutations,
      	n_hits
      );

      applyXPermutation<float>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_cos );
      //      applyXPermutation<float>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_dxDy );
      applyXPermutation<float>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_weight2 );
      applyXPermutation<float>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_xAtYEq0 );
      applyXPermutation<float>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_yBegin );
      applyXPermutation<float>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_yEnd );
      applyXPermutation<float>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_zAtYEq0 );
      applyXPermutation<unsigned int>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_LHCbID );
      applyXPermutation<int>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_planeCode );
      applyXPermutation<int>( hit_permutations, layer_offset, n_hits, hits_layers_events[i_event].m_highThreshold );
      
      for ( int i_hit = 0; i_hit < n_hits; ++i_hit ) {
	VeloUTTracking::Hit hit;
	hit.m_cos = hits_layers_events[i_event].m_cos[layer_offset + i_hit];
	//hit.m_dxDy = hits_layers_events[i_event].m_dxDy[layer_offset + i_hit];
	hit.m_weight2 = hits_layers_events[i_event].m_weight2[layer_offset + i_hit];
	hit.m_xAtYEq0 = hits_layers_events[i_event].m_xAtYEq0[layer_offset + i_hit];
	hit.m_yBegin = hits_layers_events[i_event].m_yBegin[layer_offset + i_hit];
	hit.m_yEnd = hits_layers_events[i_event].m_yEnd[layer_offset + i_hit];
	hit.m_zAtYEq0 = hits_layers_events[i_event].m_zAtYEq0[layer_offset + i_hit];
	hit.m_LHCbID = hits_layers_events[i_event].m_LHCbID[layer_offset + i_hit];
	hit.m_planeCode = hits_layers_events[i_event].m_planeCode[layer_offset + i_hit];
	hit.m_highThreshold = hits_layers_events[i_event].m_highThreshold[layer_offset + i_hit];
	
	inputHits[i_layer].push_back( hit );
	
	// For tree filling
	cos = hit.m_cos;
	yBegin = hit.m_yBegin;
	yEnd = hit.m_yEnd;
	zAtYEq0 = hit.m_zAtYEq0;
	xAtYEq0 = hit.m_xAtYEq0;
	weight2 = hit.m_weight2;
	LHCbID = hit.m_LHCbID;
	highThreshold = hit.m_highThreshold;
	layer = i_layer;
        dxDy = hit.dxDy();
        
	t_ut_hits->Fill();
      }
      // sort hits according to xAtYEq0
      //std::sort( inputHits[i_layer].begin(), inputHits[i_layer].end(), [](VeloUTTracking::Hit a, VeloUTTracking::Hit b) { return a.xAtYEq0() < b.xAtYEq0(); } );
    }
    
    // Prepare Velo tracks
    const int accumulated_tracks = host_accumulated_tracks[i_event];
    VeloState* velo_states_event = host_velo_states + accumulated_tracks;
    std::vector<VeloUTTracking::TrackVelo> tracks;
    for ( uint i_track = 0; i_track < host_number_of_tracks_pinned[i_event]; i_track++ ) {
      
      VeloUTTracking::TrackVelo track;
      
      VeloUTTracking::TrackUT ut_track;
      const uint starting_hit = host_velo_track_hit_number_pinned[accumulated_tracks + i_track];
      const uint number_of_hits = host_velo_track_hit_number_pinned[accumulated_tracks + i_track + 1] - starting_hit;
      backward = (int)(velo_states_event[i_track].backward);
      ut_track.hitsNum = number_of_hits;
      for ( int i_hit = 0; i_hit < number_of_hits; ++i_hit ) {
	ut_track.LHCbIDs.push_back( host_velo_track_hits_pinned[starting_hit + i_hit].LHCbID );
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
      const uint last_hit = starting_hit + number_of_hits - 1;
      float dx = host_velo_track_hits_pinned[last_hit].x - host_velo_track_hits_pinned[starting_hit].x;
      float dy = host_velo_track_hits_pinned[last_hit].y - host_velo_track_hits_pinned[starting_hit].y;
      float dz = host_velo_track_hits_pinned[last_hit].z - host_velo_track_hits_pinned[starting_hit].z;
      drdz = host_velo_track_hits_pinned[starting_hit].x * dx/dz + host_velo_track_hits_pinned[starting_hit].y * dy/dz;
      
      first_x = host_velo_track_hits_pinned[starting_hit].x;
      first_y = host_velo_track_hits_pinned[starting_hit].y;
      first_z = host_velo_track_hits_pinned[starting_hit].z;
      last_x = host_velo_track_hits_pinned[last_hit].x;
      last_y = host_velo_track_hits_pinned[last_hit].y;
      last_z = host_velo_track_hits_pinned[last_hit].z;
      
      t_velo_states->Fill();
      
      /* Get hits on track */
      for ( int i_hit = 0; i_hit < number_of_hits; ++i_hit ) {
	x_hit = host_velo_track_hits_pinned[starting_hit + i_hit].x;
	y_hit = host_velo_track_hits_pinned[starting_hit + i_hit].y;
	z_hit = host_velo_track_hits_pinned[starting_hit + i_hit].z;
	
	t_track_hits->Fill();
      }
      
      
      if ( backward ) continue;
      tracks.push_back( track );
    }
    //debug_cout << "at event " << i_event << ", pass " << tracks.size() << " tracks and " << inputHits[0].size() << " hits in layer 0, " << inputHits[1].size() << " hits in layer 1, " << inputHits[2].size() << " hits in layer 2, " << inputHits[3].size() << " in layer 3 to velout" << std::endl;
    
    std::vector< VeloUTTracking::TrackVeloUT > ut_tracks = velout(tracks, &(hits_layers_events[i_event]), n_hits_layers_events[i_event]);
    //debug_cout << "\t got " << (uint)ut_tracks.size() << " tracks from VeloUT " << std::endl;
    
    // store qop in tree
    for ( auto veloUT_track : ut_tracks ) {
      qop = veloUT_track.track.qop;
      t_veloUT_tracks->Fill();
    }
    
    // save in format for track checker
    std::vector< VeloUTTracking::TrackUT > ut_tracks_reduced;
    for ( auto veloUT_track : ut_tracks ) {
      ut_tracks_reduced.push_back(veloUT_track.track);
    }
    trackChecker::Tracks checker_tracks = prepareVeloUTTracks( ut_tracks_reduced );
    //debug_cout << "Passing " << checker_tracks.size() << " tracks to PrChecker" << std::endl;
    
    ut_tracks_events->emplace_back( checker_tracks );
    ut_tracks_all.push_back(ut_tracks);
    
  }
  
  
  f->Write();
  f->Close();
  
  return ut_tracks_all;
}