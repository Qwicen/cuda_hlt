#include "pv_beamline_multi_fitter.cuh"

// parameters to tune: maximum iterations, chi2max, chi2_cut, T

__global__ void pv_beamline_multi_fitter(
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks,
  PV::Vertex* dev_multi_fit_vertices,
  uint* dev_number_of_multi_fit_vertices)
{

  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  uint* number_of_multi_fit_vertices = dev_number_of_multi_fit_vertices + event_number;

  const Velo::Consolidated::Tracks velo_tracks {
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};

  const uint number_of_tracks = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  const float* zseeds = dev_zpeaks + event_number * PV::max_number_vertices;
  const uint number_of_seeds = dev_number_of_zpeaks[event_number];

  const PVTrack* tracks = dev_pvtracks + event_tracks_offset;

  PV::Vertex* vertices = dev_multi_fit_vertices + event_number * PV::max_number_vertices;

  PV::Vertex vertex;

  // make sure that we have one thread per seed
  for (uint i_thisseed = threadIdx.x; i_thisseed < number_of_seeds; i_thisseed += blockDim.x) {
    bool converged = false;
    float vtxcov[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // initial vertex posisiton, use x,y of the beamline and z of the seed
    float2 vtxpos_xy {beamline.x, beamline.y};
    auto vtxpos_z = zseeds[i_thisseed];
    const auto maxDeltaZConverged {0.001f};
    auto chi2tot = 0.f;
    unsigned short nselectedtracks = 0;
    unsigned short iter = 0;
    // debug_cout << "next vertex " << std::endl;
    for (; iter < maxFitIter && !converged; ++iter) {
      auto halfD2Chi2DX2_00 = 0.f;
      auto halfD2Chi2DX2_11 = 0.f;
      auto halfD2Chi2DX2_20 = 0.f;
      auto halfD2Chi2DX2_21 = 0.f;
      auto halfD2Chi2DX2_22 = 0.f;
      float3 halfDChi2DX {0.f, 0.f, 0.f};
      chi2tot = 0.f;
      nselectedtracks = 0;
      // debug_cout << "next track" << std::endl;
      for (int i = 0; i < number_of_tracks; i++) {
        // compute the chi2
        PVTrackInVertex trk = tracks[i];
        // skip tracks lying outside histogram range
        if (zmin > trk.z || trk.z > zmax) continue;
        const auto dz = vtxpos_z - trk.z;
        float2 res {0.f, 0.f};
        res = vtxpos_xy - (trk.x + trk.tx * dz);
        const auto chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;
        // debug_cout << "chi2 = " << chi2 << ", max = " << chi2max << std::endl;
        // compute the weight.
        trk.weight = 0.f;
        if (chi2 < maxDeltaChi2) { // to branch or not, that is the question!
                                   // if (true) {
          ++nselectedtracks;
          // for more information on the weighted fitting, see e.g.
          // Adaptive Multi-vertex fitting, R. Frühwirth, W. Waltenberger
          // https://cds.cern.ch/record/803519/files/p280.pdf
          // double T = 1. + maxNumIter / (iter+1) * 0.05;
          // float T = 1.f;

          // try out varying chi2_cut during iterations instead of T
          const auto chi2_cut = 0.1f + 0.01f * maxFitIter / (iter + 1);

          trk.weight = exp(-chi2 * 0.5f);
          auto denom = exp(-chi2_cut * 0.5f);
          for (int i_otherseed = 0; i_otherseed < number_of_seeds; i_otherseed++) {
            float2 res_otherseed {0.f, 0.f};
            const auto dz = zseeds[i_otherseed] - trk.z;

            // we calculate the residual w.r.t to the other seed positions. Since we don't update them during the fit we
            // use the beamline (x,y)
            res_otherseed = res_otherseed - (trk.x + trk.tx * dz);
            // at the moment this term reuses W matrix at z of point of closest approach -> use seed positions instead?
            const auto chi2_otherseed =
              res_otherseed.x * res_otherseed.x * trk.W_00 + res_otherseed.y * res_otherseed.y * trk.W_11;
            denom += exp(-chi2_otherseed * 0.5f);
          }
          trk.weight = trk.weight / denom;

          // unfortunately branchy, but reduces fake rate
          if (trk.weight < minWeight) continue;
          // trk.weight = sqr( 1.f - chi2 / chi2max ) ;
          // trk.weight = chi2 < 1 ? 1 : sqr( 1. - (chi2-1) / (chi2max-1) ) ;
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
          halfD2Chi2DX2_11 += trk.weight * trk.HWH_11;
          halfD2Chi2DX2_20 += trk.weight * trk.HWH_20;
          halfD2Chi2DX2_21 += trk.weight * trk.HWH_21;
          halfD2Chi2DX2_22 += trk.weight * trk.HWH_22;

          chi2tot += trk.weight * chi2;
        }
      }
      __syncthreads();

      if (nselectedtracks >= 2) {
        // compute the new vertex covariance using analytical inversion
        const auto a00 = halfD2Chi2DX2_00;
        const auto a11 = halfD2Chi2DX2_11;
        const auto a20 = halfD2Chi2DX2_20;
        const auto a21 = halfD2Chi2DX2_21;
        const auto a22 = halfD2Chi2DX2_22;

        const auto det = a00 * (a22 * a11 - a21 * a21) + a20 * (-a11 * a20);
        const auto inv_det = 1.f / det;
        // maybe we should catch the case when det = 0
        // if (det == 0) return false;

        vtxcov[0] = (a22 * a11 - a21 * a21) * inv_det;
        vtxcov[1] = -(-a20 * a21) * inv_det;
        vtxcov[2] = (a22 * a00 - a20 * a20) * inv_det;
        vtxcov[3] = (-a20 * a11) * inv_det;
        vtxcov[4] = -(a21 * a00) * inv_det;
        vtxcov[5] = (a11 * a00) * inv_det;

        // compute the delta w.r.t. the reference
        const float2 delta_xy {
          -1.f * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z),
          -1.f * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z)};

        const auto delta_z = -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z);
        chi2tot += delta_xy.x * halfDChi2DX.x + delta_xy.y * halfDChi2DX.y + delta_z * halfDChi2DX.z;

        // update the position
        vtxpos_xy = vtxpos_xy + delta_xy;
        vtxpos_z = vtxpos_z + delta_z;
        converged = std::abs(delta_z) < maxDeltaZConverged;
      }
      else {
        float3 fakepos {-99999.f, -99999.f, -99999.f};
        vertex.setPosition(fakepos);
        break;
      }
    } // end iteration loop
    // std::cout << "Number of iterations: " << iter << " " << nselectedtracks << std::endl ;
    vertex.chi2 = chi2tot;
    vertex.setPosition(vtxpos_xy, vtxpos_z);
    // vtxcov[5] = 100.;
    vertex.setCovMatrix(vtxcov);
    for (int i = 0; i < number_of_tracks; i++) {
      PVTrackInVertex trk = tracks[i];
      if (trk.weight > 0.f) vertex.n_tracks++;
    }

    // TODO integrate beamline position
    const float2 beamline {0.f, 0.f};
    const auto beamlinedx = vertex.position.x - beamline.x;
    const auto beamlinedy = vertex.position.y - beamline.y;
    const auto beamlinerho2 = beamlinedx * beamlinedx + beamlinedy * beamlinedy;
    if (vertex.n_tracks >= minNumTracksPerVertex && beamlinerho2 < maxVertexRho2) {
      uint vertex_index = atomicAdd(number_of_multi_fit_vertices, 1);
      vertices[vertex_index] = vertex;
    }
  }
}
