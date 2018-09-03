#include "run_VeloUT_CPU.h"

#ifdef WITH_ROOT
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#endif

int run_veloUT_on_CPU (
  std::vector<trackChecker::Tracks>& ut_tracks_events,
  VeloUTTracking::HitsSoA* hits_layers_events,
  const PrUTMagnetTool* host_ut_magnet_tool,
  const VeloState* host_velo_states,
  const int* host_accumulated_tracks,
  const uint* host_velo_track_hit_number_pinned,
  const VeloTracking::Hit<mc_check_enabled>* host_velo_track_hits_pinned,   
  const int* host_number_of_tracks_pinned,
  const int &number_of_events
) {

  int backward;
#ifdef WITH_ROOT
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
#endif
   
  int n_veloUT_tracks = 0;
  int n_velo_tracks_in_UT = 0;
  int n_velo_tracks = 0;
  int n_forward_velo_tracks = 0;

  uint hit_permutations[ VeloUTTracking::n_layers * VeloUTTracking::max_numhits_per_event ];
  
  for ( int i_event = 0; i_event < number_of_events; ++i_event ) {

    // Prepare hits
    VeloUTTracking::HitsSoA hits_layers = hits_layers_events[i_event];
    VeloUTTracking::HitsSoA hits_layers_sorted;
    
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      const uint layer_offset = hits_layers.layer_offset[i_layer];
      const uint n_hits = hits_layers.n_hits_layers[i_layer];

      hits_layers_sorted.n_hits_layers[i_layer] = n_hits;
      hits_layers_sorted.layer_offset[i_layer] = layer_offset;
      
      // sort according to xAtyEq0
      find_permutation<float>( 
        hits_layers.m_xAtYEq0,
        layer_offset,
      	hit_permutations,
      	n_hits
      );
          
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers.m_cos, hits_layers_sorted.m_cos );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers.m_weight, hits_layers_sorted.m_weight );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers.m_xAtYEq0, hits_layers_sorted.m_xAtYEq0 );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers.m_yBegin, hits_layers_sorted.m_yBegin );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers.m_yEnd, hits_layers_sorted.m_yEnd );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers.m_zAtYEq0, hits_layers_sorted.m_zAtYEq0 );
      apply_permutation<unsigned int>( hit_permutations, layer_offset, n_hits, hits_layers.m_LHCbID, hits_layers_sorted.m_LHCbID );
      apply_permutation<int>( hit_permutations, layer_offset, n_hits, hits_layers.m_planeCode, hits_layers_sorted.m_planeCode );
      apply_permutation<int>( hit_permutations, layer_offset, n_hits, hits_layers.m_highThreshold, hits_layers_sorted.m_highThreshold );
       
     
#ifdef WITH_ROOT
      for ( int i_hit = 0; i_hit < n_hits; ++i_hit ) {
	cos = hits_layers.m_cos[layer_offset + i_hit];
	weight = hits_layers.m_weight[layer_offset + i_hit];
	xAtYEq0 = hits_layers.m_xAtYEq0[layer_offset + i_hit];
	yBegin = hits_layers.m_yBegin[layer_offset + i_hit];
	yEnd = hits_layers.m_yEnd[layer_offset + i_hit];
	zAtYEq0 = hits_layers.m_zAtYEq0[layer_offset + i_hit];
	LHCbID = hits_layers.m_LHCbID[layer_offset + i_hit];
	layer = i_layer;
	highThreshold = hits_layers.m_highThreshold[layer_offset + i_hit];
        dxDy = hits_layers.dxDy(layer_offset + i_hit);
                
	t_ut_hits->Fill();
      }
#endif
    } // layers
  
    // Prepare Velo tracks
    const int accumulated_tracks = host_accumulated_tracks[i_event];
    const VeloState* host_velo_states_event = host_velo_states + accumulated_tracks;
    for ( uint i_track = 0; i_track < host_number_of_tracks_pinned[i_event]; i_track++ ) {

      n_velo_tracks++;
      
      const uint starting_hit = host_velo_track_hit_number_pinned[accumulated_tracks + i_track];
      const uint number_of_hits = host_velo_track_hit_number_pinned[accumulated_tracks + i_track + 1] - starting_hit;
      backward = (int)(host_velo_states_event[i_track].backward);
      if ( !backward ) n_forward_velo_tracks++;

#ifdef WITH_ROOT      
      // For tree filling
      x = host_velo_states_event[i_track].x;
      y = host_velo_states_event[i_track].y;
      tx = host_velo_states_event[i_track].tx;
      ty = host_velo_states_event[i_track].ty;
      chi2 = host_velo_states_event[i_track].chi2;
      z = host_velo_states_event[i_track].z;
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
#endif
    } // tracks

    int n_veloUT_tracks_event = 0;
    VeloUTTracking::TrackUT veloUT_tracks[VeloUTTracking::max_num_tracks];
    call_PrVeloUT(
      host_velo_track_hit_number_pinned,
      host_velo_track_hits_pinned,                                                       
      host_number_of_tracks_pinned[i_event],
      host_accumulated_tracks[i_event],
      host_velo_states_event,
      &(hits_layers_sorted),
      host_ut_magnet_tool,
      veloUT_tracks,
      n_velo_tracks_in_UT,
      n_veloUT_tracks_event
   );
    n_veloUT_tracks += n_veloUT_tracks_event;

#ifdef WITH_ROOT
    // store qop in tree
    for ( int i_track = 0; i_track < n_veloUT_tracks_event; i_track++ ) {
      qop = veloUT_tracks[i_track].qop;
      t_veloUT_tracks->Fill();
    }
#endif
    
    // save in format for track checker
    trackChecker::Tracks checker_tracks = prepareVeloUTTracksEvent( veloUT_tracks, n_veloUT_tracks_event );
    ut_tracks_events.emplace_back( checker_tracks );
    
  } // events

  info_cout << "Number of velo tracks per event = " << float(n_velo_tracks) / float(number_of_events) << std::endl;
  info_cout << "Amount of forward velo tracks = " << float(n_forward_velo_tracks) / float(n_velo_tracks) << std::endl;
  info_cout << "Amount of forward velo tracks in UT acceptance = " << float(n_velo_tracks_in_UT) / float(n_forward_velo_tracks)  << std::endl;
  info_cout << "Amount of UT tracks found ( out of velo tracks in UT acceptance ) " << float(n_veloUT_tracks) / float(n_velo_tracks_in_UT) << std::endl;

#ifdef WITH_ROOT
  f->Write();
  f->Close();
#endif
  
  return 0;
}
