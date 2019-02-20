#include <cstdio>
#include "KalmanChecker.h"

void checkKalmanTracks(
  const uint start_event_offset,
  const std::vector<trackChecker::Tracks>& tracks,
  const MCEvents selected_mc_events)
{

  // Setup the TTree.
  float trk_z, trk_x, trk_y, trk_tx, trk_ty, trk_qop;
  float trk_first_qop, trk_best_qop, trk_best_pt;
  float trk_kalman_ip, trk_kalman_ipx, trk_kalman_ipy, trk_kalman_ip_chi2;
  float trk_velo_ip, trk_velo_ipx, trk_velo_ipy, trk_velo_ip_chi2;
  float trk_chi2, trk_chi2V, trk_chi2T;
  float trk_ndof, trk_ndofV, trk_ndofT;
  float trk_ghost;
#ifdef WITH_ROOT
  TFile* outfile = new TFile("../output/KalmanIPCheckerOutput.root", "recreate");
  TTree* tree = new TTree("kalman_ip_tree", "kalman_ip_tree");
  tree->Branch("z", &trk_z);
  tree->Branch("x", &trk_x);
  tree->Branch("y", &trk_y);
  tree->Branch("tx", &trk_tx);
  tree->Branch("ty", &trk_ty);
  tree->Branch("qop", &trk_qop);
  tree->Branch("first_qop", &trk_first_qop);
  tree->Branch("best_qop", &trk_best_qop);
  tree->Branch("best_pt", &trk_best_pt);
  tree->Branch("kalman_ip", &trk_kalman_ip);
  tree->Branch("kalman_ipx", &trk_kalman_ipx);
  tree->Branch("kalman_ipy", &trk_kalman_ipy);
  tree->Branch("kalman_ip_chi2", &trk_kalman_ip_chi2);
  tree->Branch("velo_ip", &trk_velo_ip);
  tree->Branch("velo_ipx", &trk_velo_ipx);
  tree->Branch("velo_ipy", &trk_velo_ipy);
  tree->Branch("velo_ip_chi2", &trk_velo_ip_chi2);
  tree->Branch("chi2", &trk_chi2);
  tree->Branch("chi2V", &trk_chi2V);
  tree->Branch("chi2T", &trk_chi2T);
  tree->Branch("ndof", &trk_ndof);
  tree->Branch("ndofV", &trk_ndofV);
  tree->Branch("ndofT", &trk_ndofT);
  tree->Branch("ghost", &trk_ghost);
#endif

  // Loop over events.
  for (int i_event; i_event < selected_mc_events.size(); ++i_event) {

    const auto& mc_event = selected_mc_events[i_event];
    const auto& mcps = mc_event.mc_particles<TrackCheckerForward>();
    const auto& event_tracks = tracks[i_event];
    MCAssociator mcassoc {mcps};

    // Loop over tracks.
    for (auto track : event_tracks) {
      const auto& ids = track.ids();
      const auto assoc = mcassoc(ids.begin(), ids.end(), track.n_matched_total);
      if (!assoc)
        trk_ghost = 1.;
      else {
        const auto weight = assoc.front().second;
        if (weight < 0.7)
          trk_ghost = 1.;
        else
          trk_ghost = 0.;
      }
      trk_z = track.z;
      trk_x = track.x;
      trk_y = track.y;
      trk_tx = track.tx;
      trk_ty = track.ty;
      trk_qop = track.qop;
      trk_first_qop = track.first_qop;
      trk_best_qop = track.best_qop;
      trk_kalman_ip = track.kalman_ip;
      trk_kalman_ipx = track.kalman_ipx;
      trk_kalman_ipy = track.kalman_ipy;
      trk_kalman_ip_chi2 = track.kalman_ip_chi2;
      trk_velo_ip = track.velo_ip;
      trk_velo_ipx = track.velo_ipx;
      trk_velo_ipy = track.velo_ipy;
      trk_velo_ip_chi2 = track.velo_ip_chi2;
      trk_chi2 = track.chi2;
      trk_chi2V = track.chi2V;
      trk_chi2T = track.chi2T;
      trk_ndof = (float) track.ndof;
      trk_ndofV = (float) track.ndofV;
      trk_ndofT = (float) track.ndofT;
      float sint = std::sqrt((trk_tx * trk_tx + trk_ty * trk_ty) / (1. + trk_tx * trk_tx + trk_ty * trk_ty));
      trk_best_pt = sint / std::abs(track.best_qop);
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
