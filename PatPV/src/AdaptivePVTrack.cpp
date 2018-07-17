#include "../include/AdaptivePVTrack.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

//pass pointer to state?
void AdaptivePVTrack::linearTransportTo( double new_z) {
    const double dz = new_z - m_state.z ;
    const double dz2 = dz*dz ;
    m_state.x += dz * m_state.tx ;
    m_state.y += dz * m_state.ty ;
    m_state.z = new_z;
    m_state.c00 += dz2 * m_state.c22 + 2*dz* m_state.c20 ;
    m_state.c20 += dz* m_state.c22 ;
    m_state.c11 += dz2* m_state.c33 + 2* dz*m_state.c31 ;
    m_state.c31 += dz* m_state.c33 ;
    
  }


 AdaptivePVTrack::AdaptivePVTrack(VeloState& track, XYZPoint& vtx)
    
  {
    // get the state
    m_state = track ;
    

    // do here things we could evaluate at z_seed. may add cov matrix here, which'd save a lot of time.
    m_H[0] = 1 ;
    m_H[(2 * (2 + 1)) / 2 + 0] = - m_state.tx ;
    m_H[(2 * (2 + 1)) / 2 + 1] = - m_state.ty ;
    // update the cache
    updateCache( vtx ) ;
  }


  void AdaptivePVTrack::updateCache(const XYZPoint& vtx)
  {
    // transport to vtx z
    // still missing!
    std::cout << "before transport: "<< vtx.z << " " << m_state.z << std::endl;
    //std::cout << m_state.y << endl;
    linearTransportTo( vtx.z ) ;
    std::cout << "after transport: " << m_state.z << std::endl;
    //std::cout << m_state.y << endl;

    // invert cov matrix

    //write out inverse covariance matrix
    m_invcov[0] = 1. / m_state.c00;
    m_invcov[1] = 0.;
    m_invcov[2] = 1. / m_state.c11;

    // The following can all be written out, omitting the zeros, once
    // we know that it works.

    Vector2 res{ vtx.x - m_state.x, vtx.y - m_state.y };

    //do we even need HW?
    double HW[6] ;
    HW[0] = 1. / m_state.c00;
    HW[1] = 0.;
    HW[2] = 1. / m_state.c11;
    HW[3] = - m_state.tx / m_state.c00;
    HW[4] = - m_state.ty / m_state.c11;
    HW[5] = 0.;
    
    m_halfD2Chi2DX2[0] = 1. / m_state.c00;
    m_halfD2Chi2DX2[1] = 0.;
    m_halfD2Chi2DX2[2] = 1. / m_state.c11;
    m_halfD2Chi2DX2[3] = - m_state.tx / m_state.c00;
    m_halfD2Chi2DX2[4] = - m_state.ty / m_state.c11;
    m_halfD2Chi2DX2[5] = m_state.tx * m_state.tx / m_state.c00 + m_state.ty * m_state.ty / m_state.c11;

    m_halfDChi2DX.x = res.x / m_state.c00;
    m_halfDChi2DX.y = res.y / m_state.c11;
    m_halfDChi2DX.z = -m_state.tx*res.x / m_state.c00 -m_state.ty*res.y / m_state.c11;
    m_chi2          = res.x*res.x / m_state.c00 +res.y*res.y / m_state.c11;
    //    std::cout << "calculate chi2:" << std::endl;

    //std::cout << "diff x: " << vtx.x - (m_state.x ) << std::endl;
    //std::cout << "diff y: " << vtx.y - (m_state.y ) << std::endl;
    //std::cout << "err x: " << m_state.c00 << std::endl; 
    //std::cout << "err y: " << m_state.c11 << std::endl;
  }


   double AdaptivePVTrack::chi2( const XYZPoint& vtx ) const
  {
    double dz = vtx.z - m_state.z ;
    //std::cout << "calculate chi2:" << std::endl;
    //std::cout << "dz: " << dz << std::endl;
    //std::cout << "diff x: " << vtx.x - (m_state.x + dz*m_state.tx) << std::endl;
    //std::cout << "err x: " << m_state.c00 << std::endl; 
    //std::cout << "err <: " << m_state.c11 << std::endl;
    Vector2 res{ vtx.x - (m_state.x + dz*m_state.tx),
                        vtx.y - (m_state.y + dz*m_state.ty) };
    return res.x*res.x / m_state.c00 +res.y*res.y / m_state.c11;
  }