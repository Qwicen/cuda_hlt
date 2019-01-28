#include <cstdio>
#include "KalmanChecker.h"

void checkKalmanTracks(
  const uint start_event_offset,
  const std::vector<trackChecker::Tracks> &tracks,
  const MCEvents selected_mc_events
){

  // Setup the TTree.
  float trk_z, trk_x, trk_y, trk_tx, trk_ty, trk_qop;
  float trk_chi2, trk_chi2V, trk_chi2T;
  float trk_ndof, trk_ndofV, trk_ndofT;
  float trk_ghost;
#ifdef WITH_ROOT
  TFile* outfile = new TFile("../output/KalmanChecker.root","recreate");
  TTree* tree = new TTree("kf_tree", "kf_tree");
  tree->Branch("z", &trk_z);
  tree->Branch("x", &trk_x);
  tree->Branch("y", &trk_y);
  tree->Branch("tx", &trk_tx);
  tree->Branch("ty", &trk_ty);
  tree->Branch("qop", &trk_qop);
  tree->Branch("chi2", &trk_chi2);
  tree->Branch("chi2V", &trk_chi2V);
  tree->Branch("chi2T", &trk_chi2T);
  tree->Branch("ndof", &trk_ndof);
  tree->Branch("ndofV", &trk_ndofV);
  tree->Branch("ndofT", &trk_ndofT);
  tree->Branch("ghost", &trk_ghost);
#endif

  // Loop over events.
  for(int i_event; i_event < selected_mc_events.size(); ++i_event){

    const auto& mc_event = selected_mc_events[i_event];
    const auto& mcps = mc_event.mc_particles<TrackCheckerForward>();
    const auto& event_tracks = tracks[i_event];
    MCAssociator mcassoc {mcps};
    
    // Loop over tracks.
    for(auto track : event_tracks){
      const auto &ids = track.ids();
      const auto assoc = mcassoc(ids.begin(), ids.end(), track.n_matched_total);
      if(!assoc) trk_ghost = 1.;
      else{
        const auto weight = assoc.front().second;
        if(weight<0.7) trk_ghost = 1.;
        else trk_ghost = 0.;
      }
      trk_z = track.z;
      trk_x = track.x;
      trk_y = track.y;
      trk_tx = track.tx;
      trk_ty = track.ty;
      trk_qop = track.qop;
      trk_chi2 = track.chi2;
      trk_chi2V = track.chi2V;
      trk_chi2T = track.chi2T;
      trk_ndof = (float)track.ndof;
      trk_ndofV = (float)track.ndofV;
      trk_ndofT = (float)track.ndofT;
#ifdef WITH_ROOT
      tree->Fill();
#endif
    }
  }
  // Close the file.
#ifdef WITH_ROOT
  tree->Write();
  outfile->Close();
#endif
  
}
