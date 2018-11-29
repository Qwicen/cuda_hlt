#include "blpv_multi_fitter.cuh"



__global__ void blpv_multi_fitter(
  int * dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks,
  PV::Vertex* dev_multi_fit_vertices,
  uint* dev_number_of_multi_fit_vertices) {
  const uint maxNumIter = 5;
  const float chi2max= 9.f;


  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const Velo::Consolidated::Tracks velo_tracks {(uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  
  const uint number_of_tracks = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  const float * zseeds = dev_zpeaks + event_number * PV::max_number_vertices;
  const uint number_of_seeds = dev_number_of_zpeaks[event_number];

  const PVTrack* tracks = dev_pvtracks + event_tracks_offset;

  PV::Vertex* vertices = dev_multi_fit_vertices + event_number * PV::max_number_vertices;

  uint i_thisseed = threadIdx.x;
  bool converged = false ;
  float vtxcov[6];
  vtxcov[0] = 0.;
  vtxcov[1] = 0.;
  vtxcov[2] = 0.;
  vtxcov[3] = 0.;
  vtxcov[4] = 0.;
  vtxcov[5] = 0.;
  //TODO: use x,y from beamline
  float3 vtxpos = {0., 0., zseeds[i_thisseed]};
  const float maxDeltaZConverged{0.001} ;
  float chi2tot = 0;
  unsigned short nselectedtracks = 0;
  unsigned short iter = 0;
  //debug_cout << "next vertex " << std::endl;
  for(; iter<maxNumIter && !converged;++iter) {
    PV::myfloat halfD2Chi2DX2_00 = 0.;
    PV::myfloat halfD2Chi2DX2_10 = 0.;
    PV::myfloat halfD2Chi2DX2_11 = 0.;
    PV::myfloat halfD2Chi2DX2_20 = 0.;
    PV::myfloat halfD2Chi2DX2_21 = 0.;
    PV::myfloat halfD2Chi2DX2_22 = 0.;
    float3 halfDChi2DX{0.f,0.f,0.f} ;
    chi2tot = 0.f ;
    nselectedtracks = 0 ;
    float2 vtxposvec{vtxpos.x,vtxpos.y};
    //debug_cout << "next track" << std::endl;
    for( int i = 0; i < number_of_tracks; i++ ) {
      // compute the chi2
      PVTrackInVertex trk = tracks[i];
      //skip tracks lying outside histogram range
      if( m_zmin > trk.z  || trk.z > m_zmax ) continue;
      const float dz = vtxpos.z - trk.z;
      float2 res{0.f,0.f};
      res = vtxposvec - (trk.x + trk.tx*dz);
      
      float chi2 = res.x*res.x * trk.W_00 + res.y*res.y*trk.W_11 ;
     // debug_cout << "chi2 = " << chi2 << ", max = " << chi2max << std::endl;
      // compute the weight.
      trk.weight = 0 ;
      //if( chi2 < chi2max ) { // to branch or not, that is the question!
      if(true){
        ++nselectedtracks ;
        // Tukey's weight
        //double T = 1. + maxNumIter / (iter+1) * 0.05;
        double T = 1.;

        //try out varying chi2_cut during iterations instead of T
        double chi2_cut = 0.1 + 0.01*maxNumIter / (iter+1) ;

        trk.weight = exp(-chi2/2./T);
        double denom = exp(-chi2_cut/2./T);
        for (int i_otherseed = 0; i_otherseed < number_of_seeds; i_otherseed++) {
           float2 res{0.f,0.f};
           const float dz = zseeds[i_otherseed] - trk.z;
           float3 otherseedpos = {0., 0., zseeds[i_otherseed]};
           float2 otherseedvtx{otherseedpos.x,otherseedpos.y};

           res = otherseedvtx - (trk.x + trk.tx*dz);
           //at the moment this term reuses W'matrix at z of point of closest approach -> use seed positions instead?
           float chi2 = res.x*res.x * trk.W_00 + res.y*res.y*trk.W_11 ;
           denom += exp(-chi2/2./T);

        }
        trk.weight = trk.weight/denom;
        
        //trk.weight = sqr( 1.f - chi2 / chi2max ) ;


        //trk.weight = chi2 < 1 ? 1 : sqr( 1. - (chi2-1) / (chi2max-1) ) ;
        // += operator does not work for mixed FP types
        //halfD2Chi2DX2 += trk.weight * trk.HWH ;
        //halfDChi2DX   += trk.weight * trk.HW * res ;
        // if I use expressions, it crashes!
        //const Gaudi::SymMatrix3x3F thisHalfD2Chi2DX2 = weight * ROOT::Math::Similarity(H, trk.W ) ;
        float3 HWr;
        HWr.x = res.x * trk.W_00;
        HWr.y = res.y * trk.W_11;
        HWr.z = -trk.tx.x*res.x*trk.W_00 - trk.tx.y*res.y*trk.W_11;  
              
        halfDChi2DX = halfDChi2DX + HWr * trk.weight;
        
        halfD2Chi2DX2_00 += trk.weight * trk.HWH_00 ;
        halfD2Chi2DX2_10 += 0.f; 
        halfD2Chi2DX2_11 += trk.weight * trk.HWH_11 ;
        halfD2Chi2DX2_20 += trk.weight * trk.HWH_20 ;
        halfD2Chi2DX2_21 += trk.weight * trk.HWH_21 ;
        halfD2Chi2DX2_22 += trk.weight * trk.HWH_22 ;
                  
        chi2tot += trk.weight * chi2 ;
      }
    }
    if(nselectedtracks>=2) {
      // compute the new vertex covariance using analytical inversion
      PV::myfloat a00 = halfD2Chi2DX2_00;
      PV::myfloat a10 = halfD2Chi2DX2_10;
      PV::myfloat a11 = halfD2Chi2DX2_11;
      PV::myfloat a20 = halfD2Chi2DX2_20;
      PV::myfloat a21 = halfD2Chi2DX2_21;
      PV::myfloat a22 = halfD2Chi2DX2_22;
      
      PV::myfloat det = a00 * (a22 * a11 - a21 * a21) - a10 * (a22 * a10 - a21 * a20) + a20 * (a21*a10 - a11*a20);
      // if (det == 0) return false;
              
      vtxcov[0] = (a22*a11 - a21*a21) / det;
      vtxcov[1] = -(a22*a10-a20*a21) / det;
      vtxcov[2] = (a22*a00-a20*a20) / det;
      vtxcov[3] = (a21*a10-a20*a11) / det;
      vtxcov[4] = -(a21*a00-a20*a10) / det;
      vtxcov[5] = (a11*a00-a10*a10) / det;
      
      // compute the delta w.r.t. the reference
      float3 delta{0.f,0.f,0.f};
      // CHECK this
      delta.x = -1.f * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z );
      delta.y = -1.f * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z );
      delta.z = -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z );
      
      // note: this is only correct if chi2 was chi2 of reference!
      chi2tot  += delta.x * halfDChi2DX.x + delta.y * halfDChi2DX.y + delta.z * halfDChi2DX.z;
      
      // update the position
      vtxpos = vtxpos + delta;
      converged = std::abs(delta.z) < maxDeltaZConverged ;
    } else {
      PV::Vertex vertex;
      float3 fakepos{-99999.,-99999.,-99999.};
      vertex.setPosition(fakepos);
      vertices[i_thisseed] = vertex ;
      break ;
    }
  } // end iteration loop
  //std::cout << "Number of iterations: " << iter << " " << nselectedtracks << std::endl ;
  PV::Vertex vertex;
  vertex.chi2 = chi2tot ;
  vertex.setPosition(vtxpos);
  //vtxcov[5] = 100.;
  vertex.setCovMatrix(vtxcov);    
  for( int i = 0; i < number_of_tracks; i++) {
    PVTrackInVertex trk = tracks[i];
    if( trk.weight > 0 ) 
      vertex.n_tracks++;   }
  vertices[i_thisseed] = vertex ;
  
}