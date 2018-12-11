#include "blpv_multi_fitter.cuh"

// parameters to tune: maximum iterations, chi2max, chi2_cut, T

__global__ void blpv_multi_fitter(
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks,
  PV::Vertex* dev_multi_fit_vertices,
  uint* dev_number_of_multi_fit_vertices)
{
  // should tune this
  const uint maxNumIter = 2;
  const float chi2max = 9.f;

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
  for ( uint i_thisseed = threadIdx.x; i_thisseed < number_of_seeds; i_thisseed += blockDim.x) {
    bool converged = false;
    float vtxcov[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // TODO: use x,y from beamline
    float3 vtxpos = {0.f, 0.f, zseeds[i_thisseed]};
    const float maxDeltaZConverged {0.001f};
    float chi2tot = 0.f;
    unsigned short nselectedtracks = 0;
    
    unsigned short iter = 0;
    // debug_cout << "next vertex " << std::endl;
    for (; iter < maxNumIter && !converged; ++iter) {
      PV::myfloat halfD2Chi2DX2_00 = 0.f;
      PV::myfloat halfD2Chi2DX2_10 = 0.f;
      PV::myfloat halfD2Chi2DX2_11 = 0.f;
      PV::myfloat halfD2Chi2DX2_20 = 0.f;
      PV::myfloat halfD2Chi2DX2_21 = 0.f;
      PV::myfloat halfD2Chi2DX2_22 = 0.f;
      float3 halfDChi2DX {0.f, 0.f, 0.f};
      chi2tot = 0.f;
      nselectedtracks = 0;
      float2 vtxposvec {vtxpos.x, vtxpos.y};
      // debug_cout << "next track" << std::endl;
      for (int i = 0; i < number_of_tracks; i++) {
        
        // compute the chi2
        PVTrackInVertex trk = tracks[i];
        // skip tracks lying outside histogram range
        if (m_zmin > trk.z || trk.z > m_zmax) continue;
        const float dz = vtxpos.z - trk.z;
        float2 res {0.f, 0.f};
        res = vtxposvec - (trk.x + trk.tx * dz);
        float chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;
        // debug_cout << "chi2 = " << chi2 << ", max = " << chi2max << std::endl;
        // compute the weight.
        trk.weight = 0.f;
        if (chi2 < chi2max) { // to branch or not, that is the question!
                              // if (true) {
          ++nselectedtracks;
          // Tukey's weight
          // double T = 1. + maxNumIter / (iter+1) * 0.05;
          // float T = 1.f;

          // try out varying chi2_cut during iterations instead of T
          float chi2_cut = 0.1f + 0.01f * maxNumIter / (iter + 1);

          trk.weight = exp(-chi2 * 0.5f);
          float denom = exp(-chi2_cut * 0.5f);
          for (int i_otherseed = 0; i_otherseed < number_of_seeds; i_otherseed++) {
            float2 res {0.f, 0.f};
            const float dz = zseeds[i_otherseed] - trk.z;
            float2 otherseedvtx {0.f, 0.f};

            res = otherseedvtx - (trk.x + trk.tx * dz);
            // at the moment this term reuses W'matrix at z of point of closest approach -> use seed positions instead?
            float chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;
            denom += exp(-chi2 * 0.5f);
          }
          trk.weight = trk.weight / denom;

          // unfortunately branchy, but reduces fake rate
          if (trk.weight < m_minWeight) continue;
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
          halfD2Chi2DX2_10 += 0.f;
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
        PV::myfloat a00 = halfD2Chi2DX2_00;
        PV::myfloat a10 = halfD2Chi2DX2_10;
        PV::myfloat a11 = halfD2Chi2DX2_11;
        PV::myfloat a20 = halfD2Chi2DX2_20;
        PV::myfloat a21 = halfD2Chi2DX2_21;
        PV::myfloat a22 = halfD2Chi2DX2_22;

        PV::myfloat det = a00 * (a22 * a11 - a21 * a21) - a10 * (a22 * a10 - a21 * a20) + a20 * (a21 * a10 - a11 * a20);
        // maybe we should catch the case when det = 0
        // if (det == 0) return false;

        vtxcov[0] = (a22 * a11 - a21 * a21) / det;
        vtxcov[1] = -(a22 * a10 - a20 * a21) / det;
        vtxcov[2] = (a22 * a00 - a20 * a20) / det;
        vtxcov[3] = (a21 * a10 - a20 * a11) / det;
        vtxcov[4] = -(a21 * a00 - a20 * a10) / det;
        vtxcov[5] = (a11 * a00 - a10 * a10) / det;

        // compute the delta w.r.t. the reference
        float3 delta {
          -1.f * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z),
          -1.f * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z),
          -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z)
        };

        // note: this is only correct if chi2 was chi2 of reference!
        chi2tot += delta.x * halfDChi2DX.x + delta.y * halfDChi2DX.y + delta.z * halfDChi2DX.z;

        // update the position
        vtxpos = vtxpos + delta;
        converged = std::abs(delta.z) < maxDeltaZConverged;
      }
      else {
        float3 fakepos {-99999.f, -99999.f, -99999.f};
        vertex.setPosition(fakepos);
        break;
      }
    } // end iteration loop
    // std::cout << "Number of iterations: " << iter << " " << nselectedtracks << std::endl ;
    vertex.chi2 = chi2tot;
    vertex.setPosition(vtxpos);
    // vtxcov[5] = 100.;
    vertex.setCovMatrix(vtxcov);
    for (int i = 0; i < number_of_tracks; i++) {
      PVTrackInVertex trk = tracks[i];
      if (trk.weight > 0.f) vertex.n_tracks++;
    }
    
    // TODO integrate beamline position
    float2 beamline {0.f, 0.f};
    const float beamlinedx = vertex.position.x - beamline.x;
    const float beamlinedy = vertex.position.y - beamline.y;
    const float beamlinerho2 = beamlinedx * beamlinedx + beamlinedy * beamlinedy;
    if (vertex.n_tracks >= m_minNumTracksPerVertex && beamlinerho2 < m_maxVertexRho2) {
      uint vertex_index = atomicAdd(number_of_multi_fit_vertices, 1);
      vertices[vertex_index] = vertex;
    }
  }
}
