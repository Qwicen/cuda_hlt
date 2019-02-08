/** @file TrackChecker.cpp
 *
 * @brief check tracks against MC truth
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-19
 *
 * 2018-07 Dorothea vom Bruch: updated to run over different track types,
 * use exact same categories as PrChecker2,
 * take input from Renato Quagliani's TrackerDumper
 *
 * 10-12/2018 Dorothea vom Bruch: add histograms of track efficiency, ghost rate,
 * momentum resolution
 */

#include <cstdio>

#include "TrackChecker.h"

TrackChecker::~TrackChecker()
{
  std::printf(
    "%-50s: %9lu/%9lu %6.2f%% (%6.2f%%) ghosts\n",
    "TrackChecker output",
    m_nghosts,
    m_ntracks,
    100.f * float(m_nghosts) / float(m_ntracks),
    100.f * m_ghostperevent);
  m_categories.clear();
  std::printf("\n");

  // write histograms to file
#ifdef WITH_ROOT
  const std::string name = "../output/PrCheckerPlots.root";
  TFile* f = new TFile(name.c_str(), "UPDATE");
  std::string dirName = m_trackerName;
  if (m_trackerName == "VeloUT") dirName = "Upstream";
  TDirectory* trackerDir = f->mkdir(dirName.c_str());
  trackerDir->cd();
  histos.h_momentum_resolution->Write();
  histos.h_momentum_matched->Write();
  for (auto histo : histos.h_reconstructible_eta)
    histo.second->Write();
  for (auto histo : histos.h_reconstructible_p)
    histo.second->Write();
  for (auto histo : histos.h_reconstructible_pt)
    histo.second->Write();
  for (auto histo : histos.h_reconstructible_phi)
    histo.second->Write();
  for (auto histo : histos.h_reconstructible_nPV)
    histo.second->Write();
  for (auto histo : histos.h_reconstructed_eta)
    histo.second->Write();
  for (auto histo : histos.h_reconstructed_p)
    histo.second->Write();
  for (auto histo : histos.h_reconstructed_pt)
    histo.second->Write();
  for (auto histo : histos.h_reconstructed_phi)
    histo.second->Write();
  for (auto histo : histos.h_reconstructed_nPV)
    histo.second->Write();
  histos.h_ghost_nPV->Write();
  histos.h_total_nPV->Write();

  f->Write();
  f->Close();

  histos.deleteHistos(m_histo_categories);
#endif
}

void TrackChecker::TrackEffReport::operator()(const MCParticles& mcps)
{
  // find number of MCPs within category
  for (auto mcp : mcps) {
    if (m_accept(mcp)) {
      ++m_naccept, ++m_nacceptperevt;
    }
  }
}

void TrackChecker::TrackEffReport::
operator()(trackChecker::Tracks::const_reference& track, MCParticles::const_reference& mcp, const float weight)
{

  if (!m_accept(mcp)) return;
  if (!m_keysseen.count(mcp.key)) {
    ++m_nfound, ++m_nfoundperevt;
    m_keysseen.insert(mcp.key);
  }
  else {
    ++m_nclones;
  }

  // update purity
  m_hitpur *= float(m_nfound + m_nclones - 1) / float(m_nfound + m_nclones);
  m_hitpur += weight / float(m_nfound + m_nclones);
  // update hit efficiency
  auto hiteff = track.n_matched_total * weight / float(mcp.numHits);
  m_hiteff *= float(m_nfound + m_nclones - 1) / float(m_nfound + m_nclones);
  m_hiteff += hiteff / float(m_nfound + m_nclones);
}

void TrackChecker::TrackEffReport::evtEnds()
{
  m_keysseen.clear();
  if (m_nacceptperevt) {
    m_effperevt *= float(m_nevents) / float(m_nevents + 1);
    ++m_nevents;
    m_effperevt += (float(m_nfoundperevt) / float(m_nacceptperevt)) / float(m_nevents);
  }
  m_nfoundperevt = m_nacceptperevt = 0;
}

TrackChecker::TrackEffReport::~TrackEffReport()
{
  auto clonerate = 0.f, eff = 0.f;
  if (m_nfound) clonerate = float(m_nclones) / float(m_nfound + m_nfound);
  if (m_naccept) eff = float(m_nfound) / float(m_naccept);

  if (m_naccept > 0) {
    std::printf(
      "%-50s: %9lu/%9lu %6.2f%% (%6.2f%%), "
      "%9lu (%6.2f%%) clones, hit eff %6.2f%% pur %6.2f%%\n",
      m_name.c_str(),
      m_nfound,
      m_naccept,
      100.f * eff,
      100.f * m_effperevt,
      m_nclones,
      100.f * clonerate,
      100.f * m_hiteff,
      100.f * m_hitpur);
  }
}

void TrackChecker::HistoCategory::evtEnds() { m_keysseen.clear(); }

void TrackChecker::Histos::initHistos(const std::vector<HistoCategory>& histo_categories)
{
#ifdef WITH_ROOT
  // histos for efficiency
  for (auto histoCat : histo_categories) {
    const std::string& category = histoCat.m_name;
    std::string name = category + "_Eta_reconstructible";
    if (category.find("eta25") != std::string::npos) {
      h_reconstructible_eta[name] = new TH1D(name.c_str(), name.c_str(), 50, 0., 7.);
      name = category + "_Eta_reconstructed";
      h_reconstructed_eta[name] = new TH1D(name.c_str(), name.c_str(), 50, 0., 7.);
    }
    else {
      h_reconstructible_eta[name] = new TH1D(name.c_str(), name.c_str(), 100, -7., 7.);
      name = category + "_Eta_reconstructed";
      h_reconstructed_eta[name] = new TH1D(name.c_str(), name.c_str(), 100, -7., 7.);
    }
    name = category + "_P_reconstructible";
    h_reconstructible_p[name] = new TH1D(name.c_str(), name.c_str(), 50, 0., 100000.);
    name = category + "_Pt_reconstructible";
    h_reconstructible_pt[name] = new TH1D(name.c_str(), name.c_str(), 50, 0., 100000.);
    name = category + "_Phi_reconstructible";
    h_reconstructible_phi[name] = new TH1D(name.c_str(), name.c_str(), 25, -3.142, 3.142);
    name = category + "_nPV_reconstructible";
    h_reconstructible_nPV[name] = new TH1D(name.c_str(), name.c_str(), 21, -0.5, 20.5);
    name = category + "_P_reconstructed";
    h_reconstructed_p[name] = new TH1D(name.c_str(), name.c_str(), 50, 0., 100000.);
    name = category + "_Pt_reconstructed";
    h_reconstructed_pt[name] = new TH1D(name.c_str(), name.c_str(), 50, 0., 100000.);
    name = category + "_Phi_reconstructed";
    h_reconstructed_phi[name] = new TH1D(name.c_str(), name.c_str(), 25, -3.142, 3.142);
    name = category + "_nPV_reconstructed";
    h_reconstructed_nPV[name] = new TH1D(name.c_str(), name.c_str(), 21, -0.5, 20.5);
  }

  // histos for ghost rate
  h_ghost_nPV = new TH1D("nPV_Ghosts", "nPV_Ghosts", 21, -0.5, 20.5);
  h_total_nPV = new TH1D("nPV_Total", "nPV_Total", 21, -0.5, 20.5);

  // histo for momentum resolution
  h_momentum_resolution = new TH2D("dp_vs_p", "dp vs. p", 10, 0, 100000., 1000, -5., 5.);
  h_momentum_matched = new TH1D("p_matched", "p, matched", 100, 0, 100000.);
#endif
}

void TrackChecker::Histos::deleteHistos(const std::vector<HistoCategory>& histo_categories)
{
#ifdef WITH_ROOT
  for (auto histoCat : histo_categories) {
    const std::string& category = histoCat.m_name;
    std::string name = category + "_Eta_reconstructible";
    delete h_reconstructible_eta[name];
    name = category + "_Eta_reconstructed";
    delete h_reconstructed_eta[name];
    name = category + "_P_reconstructible";
    delete h_reconstructible_p[name];
    name = category + "_Pt_reconstructible";
    delete h_reconstructible_pt[name];
    name = category + "_Phi_reconstructible";
    delete h_reconstructible_phi[name];
    name = category + "_nPV_reconstructible";
    delete h_reconstructible_nPV[name];
    name = category + "_P_reconstructed";
    delete h_reconstructed_p[name];
    name = category + "_Pt_reconstructed";
    delete h_reconstructed_pt[name];
    name = category + "_Phi_reconstructed";
    delete h_reconstructed_phi[name];
    name = category + "_nPV_reconstructed";
    delete h_reconstructed_nPV[name];
  }
  delete h_ghost_nPV;
  delete h_total_nPV;
  delete h_momentum_resolution;
  delete h_momentum_matched;
#endif
}

void TrackChecker::Histos::fillReconstructibleHistos(const MCParticles& mcps, const HistoCategory& category)
{
#ifdef WITH_ROOT
  const std::string eta_name = category.m_name + "_Eta_reconstructible";
  const std::string p_name = category.m_name + "_P_reconstructible";
  const std::string pt_name = category.m_name + "_Pt_reconstructible";
  const std::string phi_name = category.m_name + "_Phi_reconstructible";
  const std::string nPV_name = category.m_name + "_nPV_reconstructible";
  for (auto mcp : mcps) {
    if (category.m_accept(mcp)) {
      h_reconstructible_eta[eta_name]->Fill(mcp.eta);
      h_reconstructible_p[p_name]->Fill(mcp.p);
      h_reconstructible_pt[pt_name]->Fill(mcp.pt);
      h_reconstructible_phi[phi_name]->Fill(mcp.phi);
      h_reconstructible_nPV[nPV_name]->Fill(mcp.nPV);
    }
  }
#endif
}

void TrackChecker::Histos::fillReconstructedHistos(const MCParticle& mcp, HistoCategory& category)
{
#ifdef WITH_ROOT
  if (!(category.m_accept(mcp))) return;
  if ((category.m_keysseen).count(mcp.key)) return; // clone track
  (category.m_keysseen).insert(mcp.key);            // not clone track, mark as matched

  const std::string eta_name = category.m_name + "_Eta_reconstructed";
  const std::string p_name = category.m_name + "_P_reconstructed";
  const std::string pt_name = category.m_name + "_Pt_reconstructed";
  const std::string phi_name = category.m_name + "_Phi_reconstructed";
  const std::string nPV_name = category.m_name + "_nPV_reconstructed";
  h_reconstructed_eta[eta_name]->Fill(mcp.eta);
  h_reconstructed_p[p_name]->Fill(mcp.p);
  h_reconstructed_pt[pt_name]->Fill(mcp.pt);
  h_reconstructed_phi[phi_name]->Fill(mcp.phi);
  h_reconstructed_nPV[nPV_name]->Fill(mcp.nPV);
#endif
}

void TrackChecker::Histos::fillTotalHistos(const MCParticle& mcp)
{
#ifdef WITH_ROOT
  h_total_nPV->Fill(mcp.nPV);
#endif
}

void TrackChecker::Histos::fillGhostHistos(const MCParticle& mcp)
{
#ifdef WITH_ROOT
  h_ghost_nPV->Fill(mcp.nPV);
#endif
}

void TrackChecker::Histos::fillMomentumResolutionHisto(const MCParticle& mcp, const float p)
{
#ifdef WITH_ROOT
  h_momentum_resolution->Fill(mcp.p, (mcp.p - p) / mcp.p);
  h_momentum_matched->Fill(mcp.p);
#endif
}

void TrackChecker::operator()(const trackChecker::Tracks& tracks, const MCAssociator& mcassoc, const MCParticles& mcps)
{
  // register MC particles
  for (auto& report : m_categories)
    report(mcps);
  // fill histograms of reconstructible MC particles in various categories
  for (auto& histo_cat : m_histo_categories) {
    histos.fillReconstructibleHistos(mcps, histo_cat);
  }

  // go through tracks
  const std::size_t ntracksperevt = tracks.size();
  std::size_t nghostsperevt = 0;
  for (auto track : tracks) {
    histos.fillTotalHistos(mcps[0]);
    // check LHCbIDs for MC association
    const auto& ids = track.ids();
    const auto assoc = mcassoc(ids.begin(), ids.end(), track.n_matched_total);
    if (!assoc) {
      ++nghostsperevt;
      histos.fillGhostHistos(mcps[0]);
      continue;
    }
    // have MC association, check weight
    const auto weight = assoc.front().second;
    if (weight < m_minweight) {
      ++nghostsperevt;
      histos.fillGhostHistos(mcps[0]);
      continue;
    }
    // okay, sufficient to proceed...
    const auto mcp = assoc.front().first;
    // add to various categories
    for (auto& report : m_categories) {
      report(track, mcp, weight);
    }
    // fill histograms of reconstructible MC particles in various categories
    for (auto& histo_cat : m_histo_categories) {
      histos.fillReconstructedHistos(mcp, histo_cat);
    }
    // fill histogram of momentum resolution
    histos.fillMomentumResolutionHisto(mcp, track.p);
  }
  // almost done, notify of end of event...
  ++m_nevents;
  for (auto& report : m_categories)
    report.evtEnds();
  for (auto& histo_cat : m_histo_categories)
    histo_cat.evtEnds();
  m_ghostperevent *= float(m_nevents - 1) / float(m_nevents);
  if (ntracksperevt) {
    m_ghostperevent += (float(nghostsperevt) / float(ntracksperevt)) / float(m_nevents);
  }
  m_nghosts += nghostsperevt, m_ntracks += ntracksperevt;
}
