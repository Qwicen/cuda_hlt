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

#include "TrackBeamLineVertexFinder.h"
#include <vector>
#include <cmath>


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

  namespace GaussApprox
  {
    constexpr int N = 2 ;
    const float a = std::sqrt(double(2*N+3) ) ;
    float integral( float x )
    {
      const float xi = x/a ;
      const float eta = 1 - xi*xi ;
      constexpr float p[] = {0.5,0.25,0.1875,0.15625} ;
      // be careful: if you choose here one order more, you also need to choose 'a' differently (a(N)=sqrt(2N+3))
      return 0.5f + xi * ( p[0] + eta * (p[1] + eta * p[2] ) )  ;
    }
  }
  
  // structure with minimal track info needed for PV search
  struct PVTrack
  {
    PVTrack() {}
    PVTrack( const Velo::State& state, double dz, unsigned short _index )
      : z{float(state.z+dz)},
  x{float(state.x+dz*state.tx),float(state.y+dz*state.ty)},
  tx{float(state.tx),float(state.ty)},index{_index}
    {
      // perhaps we should invert it /before/ switching to single FPP?
      // it doesn't seem to make much difference.
      PatPV::myfloat new_z = state.z+dz;

      PatPV::myfloat m_state_x = state.x;
      PatPV::myfloat m_state_y = state.y;
      PatPV::myfloat m_state_z = state.z;

      PatPV::myfloat m_state_tx = state.tx;
      PatPV::myfloat m_state_ty = state.ty;

    PatPV::myfloat m_state_c00 = state.c00;
    PatPV::myfloat m_state_c11 = state.c11;
    PatPV::myfloat m_state_c20 = state.c20;
    PatPV::myfloat m_state_c22 = state.c22;
    PatPV::myfloat m_state_c31 = state.c31;
    PatPV::myfloat m_state_c33 = state.c33;

    double dz2 = dz*dz ;

    m_state_x += dz * m_state_tx ;
    m_state_y += dz * m_state_ty ;
    m_state_z = new_z;
    m_state_c00 += dz2 * m_state_c22 + 2*dz* m_state_c20 ;
    m_state_c20 += dz* m_state_c22 ;
    m_state_c11 += dz2* m_state_c33 + 2* dz*m_state_c31 ;
    m_state_c31 += dz* m_state_c33 ;
    W_00 = 1. / m_state_c00;
    W_11 = 1. / m_state_c11;
    }
    float z{0} ;
    PatPV::Vector2 x ;      /// position (x,y)
    PatPV::Vector2 tx ;     /// direction (tx,ty)
    double W_00 ; /// weightmatrix
    double W_11 ;
    unsigned short index{0} ;/// index in the list with tracks
  } ;
  
  template<typename FTYPE> FTYPE sqr( FTYPE x ) { return x*x ;}

  struct Extremum
  {
    Extremum( unsigned short _index, float _value, float _integral ) :
      index{_index}, value{_value}, integral{_integral} {}
    unsigned short index;
    float value ;
    float integral ;
  } ;
  
  struct Cluster
  {
    Cluster( unsigned short _izfirst, unsigned short _izlast,  unsigned short _izmax ) :
      izfirst{_izfirst}, izlast{_izlast}, izmax{_izmax} {}
    unsigned short izfirst ;
    unsigned short izlast ;
    unsigned short izmax ;
  } ;

  struct SeedZWithIteratorPair
  {
    using iterator = std::vector< PVTrack >::iterator  ;
    float z ;
    iterator begin ;
    iterator end ;
    SeedZWithIteratorPair( float _z, iterator _begin, iterator _end) :
      z{_z},begin{_begin},end{_end} {}
  } ;
  
  // Need a small extension to the track when fitting the
  // vertex. Caching this information doesn't seem to help much
  // though.
  struct PVTrackInVertex : PVTrack
  {
    PVTrackInVertex( const Velo::State& trk )
      : PVTrack{trk}
    {
      //H matrix is symmetric and has four non-zero entries
      H_00 = 1. ;
      H_11 = 1. ;
      H_20 = - trk.tx ;
      H_21 = - trk.ty ;
      //HW: product of H and W matrices, symmetric with four non-zero entries
      HW_00 = W_00;
      HW_11 = W_11;
      HW_20 = H_20 * W_00;
      HW_21 = H_21 * W_11;
      HWH_00 = W_00;
      HWH_20 = H_20 * W_00;
      HWH_11 = W_11;
      HWH_21 = H_21 * W_11;
      HWH_22 = H_20*H_20*W_00 + H_21*H_21*W_11;
    }
    double H_00 ;
    double H_11 ;
    double H_20 ;
    double H_21 ;
    //HW: product of H and W matrices, symmetric with four non-zero entries
    double HW_00 ;
    double HW_11 ;
    double HW_20 ;
    double HW_21 ;
    double HWH_00 ;
    double HWH_20 ;
    double HWH_11 ;
    double HWH_21 ;
    double HWH_22 ;
    float weight{1} ;
  } ;


  
  // This implements the adapative vertex fit with Tukey's weights.
  PatPV::Vertex fitAdaptive( const std::vector<PVTrack>::iterator& tracksbegin,
          const std::vector<PVTrack>::iterator& tracksend,
          const PatPV::XYZPoint& seedposition,
          std::vector<unsigned short>& unusedtracks,
          unsigned short maxNumIter=5,
          float chi2max=9)
  {
    // make vector of TrackInVertex objects
    std::vector<PVTrackInVertex> tracks(tracksbegin,tracksend) ;
    bool converged = false ;

    PatPV::XYZPoint vtxpos = seedposition;

    PatPV::Vertex vertex;
    float vtxcov[6];
    vtxcov[0] = 0.;
    vtxcov[1] = 0.;
    vtxcov[2] = 0.;
    vtxcov[3] = 0.;
    vtxcov[4] = 0.;
    vtxcov[5] = 0.;

    vtxpos = seedposition;
    const float maxDeltaZConverged{0.001} ;
    double chi2tot{0} ;
    unsigned short nselectedtracks{0} ;
    unsigned short iter{0} ;
    for(; iter<maxNumIter && !converged;++iter) {
      PatPV::myfloat halfD2Chi2DX2_00 = 0.;
      PatPV::myfloat halfD2Chi2DX2_10 = 0.;
      PatPV::myfloat halfD2Chi2DX2_11 = 0.;
      PatPV::myfloat halfD2Chi2DX2_20 = 0.;
      PatPV::myfloat halfD2Chi2DX2_21 = 0.;
      PatPV::myfloat halfD2Chi2DX2_22 = 0.;
      PatPV::XYZPoint halfDChi2DX(0.,0.,0.) ;
      chi2tot = 0 ;
      nselectedtracks = 0 ;
      PatPV::Vector2 vtxposvec{float(vtxpos.x),float(vtxpos.y)} ;
      for( auto& trk : tracks ) {
  // compute the chi2
  const float dz = vtxpos.z - trk.z ;
  PatPV::Vector2 res(0.,0.);
  res.x = vtxposvec.x - (trk.x.x + dz*trk.tx.x);
  res.y = vtxposvec.y - (trk.x.y + dz*trk.tx.y);
  
  float chi2 = res.x*res.x * trk.W_00 + res.y*res.y*trk.W_11 ;
  // compute the weight.
  trk.weight = 0 ;
  if( chi2 < chi2max ) { // to branch or not, that is the question!
    ++nselectedtracks ;
    // Tukey's weight
    trk.weight = sqr( 1.f - chi2 / chi2max ) ;
    //trk.weight = chi2 < 1 ? 1 : sqr( 1. - (chi2-1) / (chi2max-1) ) ;
    // += operator does not work for mixed FP types
    //halfD2Chi2DX2 += trk.weight * trk.HWH ;
    //halfDChi2DX   += trk.weight * trk.HW * res ;
    // if I use expressions, it crashes!
    //const Gaudi::SymMatrix3x3F thisHalfD2Chi2DX2 = weight * ROOT::Math::Similarity(H, trk.W ) ;
    PatPV::XYZPoint HWr;
    HWr.x = res.x * trk.W_00;
    HWr.y = res.y * trk.W_11;
    HWr.z = -trk.tx.x*res.x*trk.W_00 - trk.tx.y*res.y*trk.W_11;

    halfDChi2DX.x += trk.weight * HWr.x ;
    halfDChi2DX.y += trk.weight * HWr.y ;
    halfDChi2DX.z += trk.weight * HWr.z ;

    halfD2Chi2DX2_00 += trk.weight * trk.HWH_00 ;
    halfD2Chi2DX2_10 += trk.weight * trk.HWH_11 ;
    halfD2Chi2DX2_11 += trk.weight * trk.HWH_11 ;
    halfD2Chi2DX2_20 += trk.weight * trk.HWH_20 ;
    halfD2Chi2DX2_21 += trk.weight * trk.HWH_21 ;
    halfD2Chi2DX2_22 += trk.weight * trk.HWH_22 ;


    chi2tot += trk.weight * chi2 ;
  }
      }
      if(nselectedtracks>=2) {
  // compute the new vertex covariance using analytical inversion
    PatPV::myfloat a00 = halfD2Chi2DX2_00;
    PatPV::myfloat a10 = halfD2Chi2DX2_10;
    PatPV::myfloat a11 = halfD2Chi2DX2_11;
    PatPV::myfloat a20 = halfD2Chi2DX2_20;
    PatPV::myfloat a21 = halfD2Chi2DX2_21;
    PatPV::myfloat a22 = halfD2Chi2DX2_22;

    PatPV::myfloat det = a00 * (a22 * a11 - a21 * a21) - a10 * (a22 * a10 - a21 * a20) + a20 * (a21*a10 - a11*a20);
   // if (det == 0) return false;


   vtxcov[0] = (a22*a11 - a21*a21) / det;
   vtxcov[1] = -(a22*a10-a20*a21) / det;
   vtxcov[2] = (a22*a00-a20*a20) / det;
   vtxcov[3] = (a21*a10-a20*a11) / det;
   vtxcov[4] = -(a21*a00-a20*a10) / det;
   vtxcov[5] = (a11*a00-a10*a10) / det;

  // compute the delta w.r.t. the reference
  PatPV::XYZPoint delta{0.,0.,0.};
    delta.x = -1.0 * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z );
    delta.y = -1.0 * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z );
    delta.z = -1.0 * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z );
  
  // note: this is only correct if chi2 was chi2 of reference!
  chi2tot  += delta.x * halfDChi2DX.x + delta.y * halfDChi2DX.y + delta.z * halfDChi2DX.z;

  // update the position
  vtxpos.x = ( vtxpos.x + delta.x ) ;
  vtxpos.y = ( vtxpos.y + delta.y ) ;
  vtxpos.z = ( vtxpos.z + delta.z ) ;
  converged = std::abs(delta.z) < maxDeltaZConverged ;
      } else {
  break ;
      }
    } // end iteration loop
    //std::cout << "Number of iterations: " << iter << " " << nselectedtracks << std::endl ;
    vertex.chi2 = chi2tot ;
    vertex.setPosition(vtxpos);
    vertex.setCovMatrix(vtxcov);    
    return vertex ;
  }




std::vector<PatPV::Vertex> findPVs(const std::vector<Velo::State>& tracks)
{
  // Get the beamline. this only accounts for position, not
  // rotation. that's something to improve! I have considered caching
  // this (with a handle for changes in the geometry, but the
  // computation is so fast that it isn't worth it.)

  // set this to (0,0) for now
  const auto beamline = PatPV::Vector2(0.f, 0.f) ;
  
  // get the tracks

  // Step 1: select tracks with velo info, compute the poca to the
  // beamline. cache the covariance matrix at this position. I'd
  // rather us a combination of copy_if and transform, but don't know
  // how to do that efficiently.
  const auto Ntrk = tracks.size() ;
  std::vector< PVTrack > pvtracks(Ntrk) ; // allocate everything upfront. don't use push_back/emplace_back
  {
    auto it = pvtracks.begin() ;
    for(short unsigned int index{0}; index<Ntrk; ++index) {
      const auto& trk = tracks[index] ;
  // compute the (chance in) z of the poca to the beam axis
  const Velo::State& s = trk ;
  const auto tx = s.tx ;
  const auto ty = s.ty ;
  const double dz = ( tx * ( beamline.x - s.x ) + ty * ( beamline.y - s.y ) ) / (tx*tx+ty*ty) ;
  const double newz = s.z + dz ;
  if( m_zmin < newz  && newz < m_zmax ) {
    *it = PVTrack{s,dz,index} ;
    ++it ;
  }
      
    }
    pvtracks.erase(it,pvtracks.end()) ;
  }
  

}






}
