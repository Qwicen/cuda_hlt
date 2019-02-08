/*****************************************************************************\
* (c) Copyright 2018 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "TrackBeamLineVertexFinder.cuh"
#include "BeamlinePVConstants.cuh"
#include "SeedZWithIteratorPair.h"
#include "FloatOperations.cuh"

#ifdef WITH_ROOT
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#endif

/** @class TrackBeamLineVertexFinder TrackBeamLineVertexFinder.cpp
 *
 * PV finding strategy:
 * step 1: select tracks with velo info and cache some information useful for PV finding
 * step 2: fill a histogram with the z of the poca to the beamline
 * step 3: do a peak search in that histogram ('vertex seeds')
 * step 4: assign tracks to the closest seed ('partitioning')
 * step 5: fit the vertices with an adapative vertex fit
 *
 *  @author Wouter Hulsbergen (Nikhef, 2018)
 **/

//=============================================================================
// ::execute()
//=============================================================================

namespace {

  namespace GaussApprox {
    constexpr int N = 2;
    const float a = std::sqrt(double(2 * N + 3));
    float integral(float x)
    {
      const float xi = x / a;
      const float eta = 1 - xi * xi;
      constexpr float p[] = {0.5, 0.25, 0.1875, 0.15625};
      // be careful: if you choose here one order more, you also need to choose 'a' differently (a(N)=sqrt(2N+3))
      return 0.5f + xi * (p[0] + eta * (p[1] + eta * p[2]));
    }
  } // namespace GaussApprox

  // This naively implements the adapative multi-vertex fit.
  void multifitAdaptive(
    const VeloState* velostates,
    const PVTrack* tracks,
    uint number_of_tracks,
    const float3* seedpositions,
    uint number_of_seeds,
    PV::Vertex* vertices,
    unsigned short maxNumIter = 5,
    float chi2max = 9.f)
  {
#ifdef WITH_ROOT
    TFile* weightfile = new TFile("weights.root", "RECREATE");
    TTree* weight_tree = new TTree("weights", "weights");
    int i_event, i_iteration, i_track;
    double b_weight;
    weight_tree->Branch("weight", &b_weight);
    weight_tree->Branch("event", &i_event);
    weight_tree->Branch("iteration", &i_iteration);
    weight_tree->Branch("nr_track", &i_track);
#endif

    // loop over all seeds, on GPU do this in parallel
    for (int i_thisseed = 0; i_thisseed < number_of_seeds; i_thisseed++) {
      bool converged = false;
      float vtxcov[6];
      vtxcov[0] = 0.f;
      vtxcov[1] = 0.f;
      vtxcov[2] = 0.f;
      vtxcov[3] = 0.f;
      vtxcov[4] = 0.f;
      vtxcov[5] = 0.f;
      float3 vtxpos = seedpositions[i_thisseed];
      const float maxDeltaZConverged {0.001};
      float chi2tot = 0;
      unsigned short nselectedtracks = 0;
      unsigned short iter = 0;
      debug_cout << "next vertex " << std::endl;
      for (; iter < maxNumIter && !converged; ++iter) {
        float halfD2Chi2DX2_00 = 0.f;
        float halfD2Chi2DX2_10 = 0.f;
        float halfD2Chi2DX2_11 = 0.f;
        float halfD2Chi2DX2_20 = 0.f;
        float halfD2Chi2DX2_21 = 0.f;
        float halfD2Chi2DX2_22 = 0.f;
        float3 halfDChi2DX {0.f, 0.f, 0.f};
        chi2tot = 0.f;
        nselectedtracks = 0;
        float2 vtxposvec {vtxpos.x, vtxpos.y};
        debug_cout << "next track" << std::endl;
        for (int i = 0; i < number_of_tracks; i++) {
          const VeloState s = velostates[i];
          // compute the (chance in) z of the poca to the beam axis
          const auto tx = s.tx;
          const auto ty = s.ty;
          // extrapolate state to seed position
          const float dz = vtxpos.z - s.z;
          PVTrackInVertex trk = PVTrack {s, dz};
          // compute the chi2
          // PVTrackInVertex trk = tracks[i];
          float2 res {0.f, 0.f};
          res = vtxposvec - (trk.x);
          double chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;
          debug_cout << "chi2 = " << chi2 << ", max = " << chi2max << std::endl;
          // compute the weight.
          trk.weight = 0;
          // probabaly no point to consider tracks with to high chi2
          // if( chi2 < chi2max ) { // to branch or not, that is the question!
          if (true) {
            ++nselectedtracks;
            // Tukey's weight
            // double T = 1. + maxNumIter / (iter+1) * 0.05;
            double T = 1.f;

            // try out varying chi2_cut during iterations instead of T
            double chi2_cut = 0.1f + 0.01f * maxNumIter / (iter + 1);
            // double chi2_cut = 16.;

            trk.weight = exp(-chi2 / 2. / T);
            double denom = exp(-chi2_cut / 2. / T);
            for (int i_otherseed = 0; i_otherseed < number_of_seeds; i_otherseed++) {
              float2 tmp_res {0.f, 0.f};
              float3 otherseedpos = seedpositions[i_otherseed];
              float2 otherseedvtx {otherseedpos.x, otherseedpos.y};
              const float dz = seedpositions[i_otherseed].z - s.z;
              PVTrackInVertex tmp_trk = PVTrack {s, dz};
              tmp_res = otherseedvtx - (tmp_trk.x);
              // at the moment this term reuses W'matrix at z of point of closest approach -> use seed positions
              // instead?
              double tmp_chi2 = tmp_res.x * tmp_res.x * tmp_trk.W_00 + tmp_res.y * tmp_res.y * tmp_trk.W_11;
              denom += exp(-tmp_chi2 / 2.f / T);
            }

            trk.weight = trk.weight / denom;
#ifdef WITH_ROOT
            i_event = i_thisseed;
            b_weight = trk.weight;
            i_iteration = iter;
            i_track = i;
            weight_tree->Fill();
#endif

            if (trk.weight < 0.3f) continue;

            float3 HWr;
            HWr.x = res.x * trk.W_00;
            HWr.y = res.y * trk.W_11;
            HWr.z = -trk.tx.x * res.x * trk.W_00 - trk.tx.y * res.y * trk.W_11;

            halfDChi2DX = halfDChi2DX + HWr * trk.weight;

            halfD2Chi2DX2_00 += trk.weight * trk.HWH_00;
            halfD2Chi2DX2_10 += 0.f;
            halfD2Chi2DX2_11 += trk.weight * trk.HWH_11;
            halfD2Chi2DX2_20 += trk.weight * trk.HWH_20;
            halfD2Chi2DX2_21 += trk.weight * trk.HWH_21;
            halfD2Chi2DX2_22 += trk.weight * trk.HWH_22;

            chi2tot += trk.weight * chi2;
          }
        }
        if (nselectedtracks >= 2) {
          // compute the new vertex covariance using analytical inversion
          float a00 = halfD2Chi2DX2_00;
          float a10 = halfD2Chi2DX2_10;
          float a11 = halfD2Chi2DX2_11;
          float a20 = halfD2Chi2DX2_20;
          float a21 = halfD2Chi2DX2_21;
          float a22 = halfD2Chi2DX2_22;

          float det = a00 * (a22 * a11 - a21 * a21) - a10 * (a22 * a10 - a21 * a20) + a20 * (a21 * a10 - a11 * a20);
          // if (det == 0) return false;

          vtxcov[0] = (a22 * a11 - a21 * a21) / det;
          vtxcov[1] = -(a22 * a10 - a20 * a21) / det;
          vtxcov[2] = (a22 * a00 - a20 * a20) / det;
          vtxcov[3] = (a21 * a10 - a20 * a11) / det;
          vtxcov[4] = -(a21 * a00 - a20 * a10) / det;
          vtxcov[5] = (a11 * a00 - a10 * a10) / det;

          // compute the delta w.r.t. the reference
          float3 delta {0.f, 0.f, 0.f};
          // CHECK this
          delta.x = -1.f * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z);
          delta.y = -1.f * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z);
          delta.z = -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z);

          // note: this is only correct if chi2 was chi2 of reference!
          chi2tot += delta.x * halfDChi2DX.x + delta.y * halfDChi2DX.y + delta.z * halfDChi2DX.z;

          // update the position
          vtxpos = vtxpos + delta;
          converged = std::abs(delta.z) < maxDeltaZConverged;
        }
        else {
          PV::Vertex vertex;
          float3 fakepos {-99999.f, -99999.f, -99999.f};
          vertex.setPosition(fakepos);
          vertices[i_thisseed] = vertex;
          break;
        }
      } // end iteration loop
      // std::cout << "Number of iterations: " << iter << " " << nselectedtracks << std::endl ;
      PV::Vertex vertex;
      vertex.chi2 = chi2tot;
      vertex.setPosition(vtxpos);
      // vtxcov[5] = 100.;
      vertex.setCovMatrix(vtxcov);
      for (int i = 0; i < number_of_tracks; i++) {
        PVTrackInVertex trk = tracks[i];
        if (trk.weight > 0) vertex.n_tracks++;
      }
      vertices[i_thisseed] = vertex;
    }
#ifdef WITH_ROOT
    weight_tree->Write();
    weightfile->Close();
#endif
  }

  // This implements the adapative vertex fit with Tukey's weights.
  PV::Vertex fitAdaptive(
    const PVTrack* tracks,
    uint number_of_tracks,
    const float3& seedposition,
    unsigned short maxNumIter = 5,
    float chi2max = 9.f)
  {
    // make vector of TrackInVertex objects
    bool converged = false;

    float3 vtxpos = seedposition;

    PV::Vertex vertex;
    float vtxcov[6];
    vtxcov[0] = 0.f;
    vtxcov[1] = 0.f;
    vtxcov[2] = 0.f;
    vtxcov[3] = 0.f;
    vtxcov[4] = 0.f;
    vtxcov[5] = 0.f;

    const float maxDeltaZConverged {0.001f};
    float chi2tot = 0.f;
    unsigned short nselectedtracks = 0;
    unsigned short iter = 0;
    debug_cout << "next vertex " << std::endl;
    for (; iter < maxNumIter && !converged; ++iter) {
      float halfD2Chi2DX2_00 = 0.f;
      float halfD2Chi2DX2_10 = 0.f;
      float halfD2Chi2DX2_11 = 0.f;
      float halfD2Chi2DX2_20 = 0.f;
      float halfD2Chi2DX2_21 = 0.f;
      float halfD2Chi2DX2_22 = 0.f;
      float3 halfDChi2DX {0.f, 0.f, 0.f};
      chi2tot = 0.f;
      nselectedtracks = 0;
      float2 vtxposvec {vtxpos.x, vtxpos.y};
      debug_cout << "next track" << std::endl;
      for (int i = 0; i < number_of_tracks; i++) {
        // compute the chi2
        PVTrackInVertex trk = tracks[i];
        const float dz = vtxpos.z - trk.z;
        float2 res {0.f, 0.f};
        res = vtxposvec - (trk.x + trk.tx * dz);

        float chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;
        debug_cout << "chi2 = " << chi2 << ", max = " << chi2max << std::endl;
        // compute the weight.
        trk.weight = 0;
        if (chi2 < chi2max) { // to branch or not, that is the question!
          ++nselectedtracks;
          // Tukey's weight
          trk.weight = 1.f - chi2 / chi2max;
          trk.weight = trk.weight * trk.weight;
          // += operator does not work for mixed FP types
          // halfD2Chi2DX2 += trk.weight * trk.HWH ;
          // halfDChi2DX   += trk.weight * trk.HW * res ;
          // if I use expressions, it crashes!
          // const Gaudi::SymMatrix3x3F thisHalfD2Chi2DX2 = weight * ROOT::Math::Similarity(H, trk.W ) ;
          float3 HWr;
          HWr.x = res.x * trk.W_00;
          HWr.y = res.y * trk.W_11;
          HWr.z = -trk.tx.x * res.x * trk.W_00 - trk.tx.y * res.y * trk.W_11;

          halfDChi2DX = halfDChi2DX + HWr * trk.weight;

          halfD2Chi2DX2_00 += trk.weight * trk.HWH_00;
          halfD2Chi2DX2_10 += 0.f;
          halfD2Chi2DX2_11 += trk.weight * trk.HWH_11;
          halfD2Chi2DX2_20 += trk.weight * trk.HWH_20;
          halfD2Chi2DX2_21 += trk.weight * trk.HWH_21;
          halfD2Chi2DX2_22 += trk.weight * trk.HWH_22;

          chi2tot += trk.weight * chi2;
        }
      }
      if (nselectedtracks >= 2) {
        // compute the new vertex covariance using analytical inversion
        float a00 = halfD2Chi2DX2_00;
        float a10 = halfD2Chi2DX2_10;
        float a11 = halfD2Chi2DX2_11;
        float a20 = halfD2Chi2DX2_20;
        float a21 = halfD2Chi2DX2_21;
        float a22 = halfD2Chi2DX2_22;

        float det = a00 * (a22 * a11 - a21 * a21) - a10 * (a22 * a10 - a21 * a20) + a20 * (a21 * a10 - a11 * a20);
        // if (det == 0) return false;

        vtxcov[0] = (a22 * a11 - a21 * a21) / det;
        vtxcov[1] = -(a22 * a10 - a20 * a21) / det;
        vtxcov[2] = (a22 * a00 - a20 * a20) / det;
        vtxcov[3] = (a21 * a10 - a20 * a11) / det;
        vtxcov[4] = -(a21 * a00 - a20 * a10) / det;
        vtxcov[5] = (a11 * a00 - a10 * a10) / det;

        // compute the delta w.r.t. the reference
        float3 delta {0.f, 0.f, 0.f};
        // CHECK this
        delta.x = -1.f * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z);
        delta.y = -1.f * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z);
        delta.z = -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z);

        // note: this is only correct if chi2 was chi2 of reference!
        chi2tot += delta.x * halfDChi2DX.x + delta.y * halfDChi2DX.y + delta.z * halfDChi2DX.z;

        // update the position
        vtxpos = vtxpos + delta;
        converged = std::abs(delta.z) < maxDeltaZConverged;
      }
      else {
        break;
      }
    } // end iteration loop
    // std::cout << "Number of iterations: " << iter << " " << nselectedtracks << std::endl ;
    vertex.chi2 = chi2tot;
    vertex.setPosition(vtxpos);
    vertex.setCovMatrix(vtxcov);
    for (int i = 0; i < number_of_tracks; i++) {
      PVTrackInVertex trk = tracks[i];
      if (trk.weight > 0) vertex.n_tracks++;
    }
    return vertex;
  }

} // namespace

void findPVs(
  char* kalmanvelo_states,
  int* velo_atomics,
  uint* velo_track_hit_number,
  PV::Vertex* reconstructed_pvs,
  int* number_of_pvs,
  const uint number_of_events)
{

#ifdef WITH_ROOT
  // Histograms only for checking and debugging
  TFile* f = new TFile("../output/PVs.root", "RECREATE");
  // TTree *t_velo_states = new TTree("velo_states", "velo_states");
  TTree* t_velo_states = new TTree("velo_states", "velo_states");
  double cov_x, cov_y, cov_z;
  float tx, ty, x, y, z;
  t_velo_states->Branch("cov_x", &cov_x);
  t_velo_states->Branch("cov_y", &cov_y);
  t_velo_states->Branch("cov_z", &cov_z);
  t_velo_states->Branch("x", &x);
  t_velo_states->Branch("y", &y);
  t_velo_states->Branch("z", &z);
  t_velo_states->Branch("tx", &tx);
  t_velo_states->Branch("ty", &ty);
  TH1F* h_z0[number_of_events];
  TH1F* h_vx[number_of_events];
  TH1F* h_vy[number_of_events];
  TH1F* h_vz[number_of_events];
  for (int i = 0; i < number_of_events; ++i) {
    std::string name = "z0_" + std::to_string(i);
    h_z0[i] = new TH1F(name.c_str(), "", Nbins, 0, Nbins - 1);
    name = "vx_" + std::to_string(i);
    h_vx[i] = new TH1F(name.c_str(), "", 100, -1, 1);
    name = "vy_" + std::to_string(i);
    h_vy[i] = new TH1F(name.c_str(), "", 100, -1, 1);
    name = "vz_" + std::to_string(i);
    h_vz[i] = new TH1F(name.c_str(), "", 100, -300, 300);
  }
  // t_z0->Branch("z0", &z0, "z0[number_of_events]/F");
#endif

  for (uint event_number = 0; event_number < number_of_events; event_number++) {
    debug_cout << "AT EVENT " << event_number << std::endl;
    int& n_pvs = number_of_pvs[event_number];
    n_pvs = 0;

    // get consolidated states
    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) velo_atomics, velo_track_hit_number, event_number, number_of_events};
    const Velo::Consolidated::States velo_states =
      Velo::Consolidated::States(kalmanvelo_states, velo_tracks.total_number_of_tracks);
    const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
    const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

    // Step 1: select tracks with velo info, compute the poca to the
    // beamline. cache the covariance matrix at this position. I'd
    // rather us a combination of copy_if and transform, but don't know
    // how to do that efficiently.
    const auto Ntrk = number_of_tracks_event; // tracks.size() ;
    debug_cout << "# of input velo states: " << Ntrk << std::endl;
    std::vector<PVTrack> pvtracks_old(Ntrk); // allocate everything upfront. don't use push_back/emplace_back
    PVTrack pvtracks[Ntrk];
    VeloState event_velo_states[Ntrk];
    // only use tracks within a certain z-range
    uint number_of_tracks_in_zrange = 0;

    {
      auto it = pvtracks_old.begin();
      for (short unsigned int index = 0; index < Ntrk; ++index) {
        const VeloState s = velo_states.get(event_tracks_offset + index);
        // compute the (chance in) z of the poca to the beam axis
        const auto tx = s.tx;
        const auto ty = s.ty;
        const float dz = (tx * (beamline.x - s.x) + ty * (beamline.y - s.y)) / (tx * tx + ty * ty);
        const double newz = s.z + dz;
        if (zmin < newz && newz < zmax) {
          pvtracks[number_of_tracks_in_zrange] = PVTrack {s, dz};
          event_velo_states[number_of_tracks_in_zrange] = s;
          number_of_tracks_in_zrange++;
          *it = PVTrack {s, dz};
          ++it;
        }
      }
    }

    debug_cout << "Selected " << (float) (number_of_tracks_in_zrange) / Ntrk << " states for PV seeds " << std::endl;

    // Step 2: fill a histogram with the z position of the poca. Use the
    // projected vertex error on that position as the width of a
    // gauss. Divide the gauss properly over the bins. This is quite
    // slow: some simplification may help here.

    // we need to define what a bin is: integral between
    //   zmin + ibin*dz and zmin + (ibin+1)*dz
    // we'll have lot's of '0.5' in the code below. at some point we may
    // just want to shift the bins.

    // this can be changed into an std::accumulate

    // std::vector<float> zhisto(Nbins,0.0f) ;
    float zhisto[Nbins] = {0.f};
    {
      for (int i = 0; i < number_of_tracks_in_zrange; i++) {
        PVTrack trk = pvtracks[i];
#ifdef WITH_ROOT
        cov_x = trk.W_00;
        cov_y = trk.W_11;
        cov_z = 1.;
        x = trk.x.x;
        y = trk.x.y;
        z = trk.z;
        tx = trk.tx.x;
        ty = trk.tx.y;
        t_velo_states->Fill();
#endif
        // bin in which z0 is, in floating point
        const float zbin = (trk.z - zmin) / dz;

        // to compute the size of the window, we use the track
        // errors. eventually we can just parametrize this as function of
        // track slope.
        const float zweight = trk.tx.x * trk.tx.x * trk.W_00 + trk.tx.y * trk.tx.y * trk.W_11;
        const float zerr = 1 / std::sqrt(zweight);
        // get rid of useless tracks. must be a bit carefull with this.
        if (zerr < maxTrackZ0Err) { // m_nsigma < 10*m_dz ) {
          const float halfwindow = GaussApprox::a * zerr / dz;
          // this looks a bit funny, but we need the first and last bin of the histogram to remain empty.
          const int minbin = std::max(int(zbin - halfwindow), 1);
          const int maxbin = std::min(int(zbin + halfwindow), Nbins - 2);
          // we can get rid of this if statement if we make a selection of seeds earlier
          if (maxbin >= minbin) {
            float integral = 0.f;
            for (auto i = minbin; i < maxbin; ++i) {
              const float relz = (zmin + (i + 1) * dz - trk.z) / zerr;
              const float thisintegral = GaussApprox::integral(relz);
              zhisto[i] += thisintegral - integral;
              integral = thisintegral;
            }
            // deal with the last bin
            zhisto[maxbin] += 1.f - integral;
          }
        }
      }
    }
#ifdef WITH_ROOT
    for (int i = 0; i < Nbins; ++i) {
      h_z0[event_number]->SetBinContent(i, zhisto[i]);
    }
#endif

    // Step 3: perform a peak search in the histogram. This used to be
    // very simple but the logic needed to find 'significant dips' made
    // it a bit more complicated. In the end it doesn't matter so much
    // because it takes relatively little time.

    // FIXME: the logic is a bit too complicated here. need to see if we
    // simplify something without loosing efficiency.
    // std::vector<Cluster> clusters ;
    //&&( zhisto[i] > zhisto[i-2] || zhisto[i] > zhisto[i+2])
    Cluster clusters[PV::max_number_of_clusters];
    uint number_of_clusters = 0;
    /*
        //try to find a simpler peak finding, the numbers here could be optimized
        for(uint i = 2; i < Nbins-2; i++) {
          if(zhisto[i] > zhisto[i -1] && zhisto[i] > zhisto[i+1] && (zhisto[i] + zhisto[i-1] + zhisto[i+1]+ zhisto[i-2]
       + zhisto[i+2] > 2.5 ) && zhisto[i] > 1.5 ) { clusters[number_of_clusters] = Cluster(i-1, i,i+1); std::cout <<
       "cluster " << i *m_dz + m_zmin << " " << zhisto[i-1] << " " << zhisto[i] << " " << zhisto[i+1] << std::endl;
            number_of_clusters++;
          }
        }
    */
    {
      // step A: make 'ProtoClusters'
      // Step B: for each such ProtoClusters
      //    - find the significant extrema (an odd number, start with a minimum. you can always achieve this by adding a
      //    zero bin at the beginning)
      //      an extremum is a bin-index, plus the integral till that point, plus the content of the bin
      //    - find the highest extremum and
      //       - try and partition at the lowest minimum besides it
      //       - if that doesn't work, try the other extremum
      //       - if that doesn't work, accept as cluster

      // Step A: make 'proto-clusters': these are subsequent bins with non-zero content and an integral above the
      // threshold.

      using BinIndex = unsigned short;
      BinIndex clusteredges[PV::max_number_clusteredges];
      uint number_of_clusteredges = 0;
      {
        const float threshold = dz / (10.f * maxTrackZ0Err); // need something sensible that depends on binsize
        bool prevempty = true;
        float integral = zhisto[0];
        for (BinIndex i = 1; i < Nbins; ++i) {
          integral += zhisto[i];
          bool empty = zhisto[i] < threshold;
          if (empty != prevempty) {
            if (prevempty || integral > minTracksInSeed) {
              clusteredges[number_of_clusteredges] = i;
              number_of_clusteredges++;
            }
            else
              number_of_clusteredges--;
            prevempty = empty;
            integral = 0;
          }
        }
      }
      debug_cout << "Found " << number_of_clusteredges / 2 << " proto clusters" << std::endl;

      // Step B: turn these into clusters. There can be more than one cluster per proto-cluster.
      const size_t Nproto = number_of_clusteredges / 2;
      for (unsigned short i = 0; i < Nproto; ++i) {
        const BinIndex ibegin = clusteredges[i * 2];
        const BinIndex iend = clusteredges[i * 2 + 1];
        // std::cout << "Trying cluster: " << ibegin << " " << iend << std::endl ;

        // find the extrema
        const float mindip = minDipDensity * dz; // need to invent something
        const float minpeak = minDensity * dz;

        // std::vector<Extremum> extrema ;
        Extremum extrema[PV::max_number_vertices];
        uint number_of_extrema = 0;
        {
          bool rising = true;
          float integral = zhisto[ibegin];
          extrema[number_of_extrema] = Extremum(ibegin, zhisto[ibegin], integral);
          number_of_extrema++;
          for (unsigned short i = ibegin; i < iend; ++i) {
            const auto value = zhisto[i];
            bool stillrising = zhisto[i + 1] > value;
            if (rising && !stillrising && value >= minpeak) {
              const auto n = number_of_extrema;
              if (n >= 2) {
                // check that the previous mimimum was significant. we
                // can still simplify this logic a bit.
                const auto dv1 = extrema[n - 2].value - extrema[n - 1].value;
                // const auto di1 = extrema[n-1].index - extrema[n-2].index ;
                const auto dv2 = value - extrema[n - 1].value;
                if (dv1 > mindip && dv2 > mindip) {
                  extrema[number_of_extrema] = Extremum(i, value, integral + 0.5f * value);
                  number_of_extrema++;
                }
                else if (dv1 > dv2) {
                  number_of_extrema--;
                  if (number_of_extrema < 0) number_of_extrema = 0;
                }
                else {
                  number_of_extrema--;
                  number_of_extrema--;
                  if (number_of_extrema < 0) number_of_extrema = 0;
                  extrema[number_of_extrema] = Extremum(i, value, integral + 0.5f * value);
                  number_of_extrema++;
                }
              }
              else {
                extrema[number_of_extrema] = Extremum(i, value, integral + 0.5f * value);
                number_of_extrema++;
              }
            }
            else if (rising != stillrising) {
              extrema[number_of_extrema] = Extremum(i, value, integral + 0.5f * value);
              number_of_extrema++;
            }
            rising = stillrising;
            integral += value;
          }
          assert(rising == false);
          extrema[number_of_extrema] = Extremum(iend, zhisto[iend], integral);
          number_of_extrema++;
        }

        // now partition on  extrema
        const auto N = number_of_extrema;
        // std::vector<Cluster> subclusters ;
        Cluster subclusters[PV::max_number_subclusters];
        uint number_of_subclusters = 0;
        if (N > 3) {
          for (unsigned int i = 1; i < N / 2 + 1; ++i) {
            if (extrema[2 * i].integral - extrema[2 * i - 2].integral > minTracksInSeed) {
              subclusters[number_of_subclusters] =
                Cluster(extrema[2 * i - 2].index, extrema[2 * i].index, extrema[2 * i - 1].index);
              number_of_subclusters++;
            }
          }
        }
        if (number_of_subclusters == 0) {
          // FIXME: still need to get the largest maximum!
          if (extrema[1].value >= minpeak) {
            clusters[number_of_clusters] =
              Cluster(extrema[0].index, extrema[number_of_extrema - 1].index, extrema[1].index);
            number_of_clusters++;
          }
        }
        else {
          // adjust the limit of the first and last to extend to the entire protocluster
          subclusters[0].izfirst = ibegin;
          subclusters[number_of_subclusters].izlast = iend;
          for (int i = 0; i < number_of_subclusters; i++) {
            Cluster subcluster = subclusters[i];
            clusters[number_of_clusters] = subcluster;
            number_of_clusters++;
          }
        }
      }
    }

    debug_cout << "Found " << number_of_clusters << " clusters" << std::endl;

    // Step 4: partition the set of tracks by vertex seed: just
    // choose the closest one. The easiest is to loop over tracks and
    // assign to closest vertex by looping over all vertices. However,
    // that becomes very slow as time is proportional to both tracks and
    // vertices. A better method is to rely on the fact that vertices
    // are sorted in z, and then use std::partition, to partition the
    // track list on the midpoint between two vertices. The logic is
    // slightly complicated to deal with partitions that have too few
    // tracks. I checked it by comparing to the 'slow' method.

    // I found that this funny weighted 'maximum' is better than most other inexpensive solutions.
    auto zClusterMean = [&zhisto](auto izmax) -> float {
      const float* b = zhisto + izmax;
      float d1 = *b - *(b - 1);
      float d2 = *b - *(b + 1);
      float idz = d1 + d2 > 0 ? 0.5f * (d1 - d2) / (d1 + d2) : 0.0f;
      return zmin + dz * (izmax + idz + 0.5f);
    };

    // std::vector<SeedZWithIteratorPair> seedsZWithIteratorPair ;
    SeedZWithIteratorPair seedsZWithIteratorPair[number_of_clusters];
    uint number_of_seedsZWIP = 0;

    if (number_of_clusters != 0) {
      std::vector<PVTrack>::iterator it = pvtracks_old.begin();
      int iprev = 0;
      for (int i = 0; i < int(number_of_clusters) - 1; ++i) {
        // const float zmid = 0.5f*(zseeds[i+1].z+zseeds[i].z) ;
        const float zmid = zmin + dz * 0.5f * (clusters[i].izlast + clusters[i + 1].izfirst + 1.f);
        std::vector<PVTrack>::iterator newit =
          std::partition(it, pvtracks_old.end(), [zmid](const auto& trk) { return trk.z < zmid; });
        // complicated logic to get rid of partitions that are too small, doign the least amount of work
        if (std::distance(it, newit) >= minNumTracksPerVertex) {
          seedsZWithIteratorPair[number_of_seedsZWIP] =
            SeedZWithIteratorPair(zClusterMean(clusters[i].izmax), it, newit);
          number_of_seedsZWIP++;
          iprev = i;
        }
        else {
          // if the partition is too small, then repartition the stuff we
          // have just isolated and assign to the previous and next. You
          // could also 'skip' this partition, but then you do too much
          // work for the next.
          if (number_of_seedsZWIP != 0 && newit != it) {
            const float zmid = zmin + dz * (clusters[iprev].izlast + clusters[i + 1].izfirst + 0.5f);
            newit = std::partition(it, newit, [zmid](const auto& trk) { return trk.z < zmid; });
            // update the last one
            seedsZWithIteratorPair[number_of_seedsZWIP - 1].end = newit;
          }
        }
        it = newit;
      }
      // Make sure to add the last partition
      if (std::distance(it, pvtracks_old.end()) >= minNumTracksPerVertex) {
        seedsZWithIteratorPair[number_of_seedsZWIP] =
          SeedZWithIteratorPair(zClusterMean(clusters[number_of_clusters - 1].izmax), it, pvtracks_old.end());
        number_of_seedsZWIP++;
      }
      else if (number_of_seedsZWIP != 0) {
        seedsZWithIteratorPair[number_of_seedsZWIP - 1].end = pvtracks_old.end();
      }
    }

    for (int i = 0; i < number_of_seedsZWIP; i++) {
      SeedZWithIteratorPair seed = seedsZWithIteratorPair[i];
      debug_cout << "Associated " << seed.end - seed.begin << " tracks to seed " << std::endl;
    }

    // Step 5: perform the adaptive vertex fit for each seed.
    // PV::Vertex preselected_vertices[PV::max_number_vertices];
    uint number_preselected_vertices = 0;
    float3 seed_positions[number_of_seedsZWIP];
    for (int i = 0; i < number_of_seedsZWIP; i++) {
      seed_positions[i] = float3 {beamline.x, beamline.y, seedsZWithIteratorPair[i].z};
    }
    PV::Vertex preselected_vertices[number_of_seedsZWIP];
    multifitAdaptive(
      event_velo_states,
      pvtracks,
      number_of_tracks_in_zrange,
      seed_positions,
      number_of_seedsZWIP,
      preselected_vertices);

    number_preselected_vertices = number_of_seedsZWIP;
    /*
       for ( int i = 0; i < number_of_seedsZWIP; i++ ) {
         SeedZWithIteratorPair seed = seedsZWithIteratorPair[i];
         PV::Vertex vertex = fitAdaptive(seed.get_array(),seed.get_size(),
                                         float3{beamline.x,beamline.y,seed.z},
                                         m_maxFitIter,m_maxDeltaChi2) ;
         preselected_vertices[number_preselected_vertices] = vertex;
         number_preselected_vertices++;
       }

       debug_cout << "Vertices remaining after fitter: " << number_preselected_vertices << std::endl;

       for ( int i = 0; i < number_preselected_vertices; i++ ) {
         PV::Vertex vertex = preselected_vertices[i];
         debug_cout << "   vertex has " << vertex.n_tracks << " tracks, x = " << vertex.position.x << ", y = " <<
       vertex.position.y << ", z = " << vertex.position.z << std::endl;
       }
   */
    // Steps that we could still take:
    // * remove vertices with too little tracks
    // * assign unused tracks to other vertices
    // * merge vertices that are close

    // create the output container
    const auto maxVertexRho2 = maxVertexRho * maxVertexRho;
    for (int i = 0; i < number_preselected_vertices; i++) {
      PV::Vertex vertex = preselected_vertices[i];

      const auto beamlinedx = vertex.position.x - beamline.x;
      const auto beamlinedy = vertex.position.y - beamline.y;
      const auto beamlinerho2 = beamlinedx * beamlinedx + beamlinedy * beamlinedy;
#ifdef WITH_ROOT
      h_vx[event_number]->Fill(vertex.position.x);
      h_vy[event_number]->Fill(vertex.position.y);
      h_vz[event_number]->Fill(vertex.position.z);
#endif
      if (vertex.n_tracks >= minNumTracksPerVertex && beamlinerho2 < maxVertexRho2) {
        reconstructed_pvs[PV::max_number_vertices * event_number + n_pvs++] = vertex;
      }
    }

  } // event loop

#ifdef WITH_ROOT
  f->Write();
  f->Close();
#endif
}
