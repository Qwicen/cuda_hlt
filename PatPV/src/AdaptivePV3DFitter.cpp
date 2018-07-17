#include <vector>
#include <cmath>
#include <iostream>

#include "../include/AdaptivePV3DFitter.h"
#include "../../PrVeloUT/include/CholeskyDecomp.h"




//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
AdaptivePV3DFitter::AdaptivePV3DFitter()
  
{
  m_trackChi = std::sqrt(m_trackMaxChi2);
}



//=============================================================================
// Least square adaptive fitting method
//=============================================================================
bool AdaptivePV3DFitter::fitVertex( XYZPoint& seedPoint,
              VeloState * host_velo_states,
             Vertex& vtx,
             std::vector<VeloState>& tracks2remove, int number_of_tracks) 
{
  //tracks2remove.clear();

  // position at which derivatives are evaluated
  XYZPoint refpos = seedPoint ;

  // prepare tracks
  std::vector<AdaptivePVTrack> pvTracks ;
  pvTracks.reserve( number_of_tracks ) ;
  //for( const auto& track : rTracks ) {
  for(int i = 0; i < number_of_tracks; i++) {  
      pvTracks.emplace_back( host_velo_states[i], refpos );
      //std::cout << "pos state x: " << track->position().x << " " << refpos.x << std::endl;
      //std::cout << "pos state y: " << track->position().y << " " << refpos.y << std::endl;
      //std::cout << "pos state z: " << track->position().z << " " << refpos.z << std::endl;
      //std::cout << "chi2: " << pvTracks.back().chi2() << " " << m_maxChi2 << std::endl;
      if (pvTracks.back().chi2() >= m_maxChi2) pvTracks.pop_back();
  }
    

  if( pvTracks.size() < m_minTr ) {
    //std::cout << pvTracks.size() << " " << m_minTr << std::endl;
    std::cout << "Too few tracks to fit PV" << std::endl;
    return false;
    }

  // current vertex position
  XYZPoint vtxpos = refpos ;
  // vertex covariance matrix
  double vtxcov[6] ;
  bool converged = false;
  double maxdz = m_maxDeltaZ;
  int nbIter = 0;
  while( (nbIter < m_minIter) || (!converged && nbIter < m_Iterations) )
  {
    ++nbIter;

    double halfD2Chi2DX2[6] = {0.,0.,0.,0.,0.,0.};
    XYZPoint halfDChi2DX(0.,0.,0.) ;
    
    // update cache if too far from reference position. this is the slow part.
    if( std::abs(refpos.z - vtxpos.z > m_maxDeltaZCache) ) {
      refpos = vtxpos ;
      for( auto& trk : pvTracks ) trk.updateCache( refpos ) ;
    };

    // add contribution from all tracks
    double chi2(0) ;
    size_t ntrin(0) ;
    for( auto& trk : pvTracks ) {
      // compute weight
      double trkchi2 = trk.chi2(vtxpos) ;
      double weight = getTukeyWeight(trkchi2, nbIter) ;
      trk.setWeight(weight) ;
      // add the track
      if ( weight > m_minTrackWeight ) {
        ++ntrin;
        halfD2Chi2DX2[0] += weight * trk.halfD2Chi2DX2()[0] ;
        halfD2Chi2DX2[1] += weight * trk.halfD2Chi2DX2()[1] ;
        halfD2Chi2DX2[2] += weight * trk.halfD2Chi2DX2()[2] ;
        halfD2Chi2DX2[3] += weight * trk.halfD2Chi2DX2()[3] ;
        halfD2Chi2DX2[4] += weight * trk.halfD2Chi2DX2()[4] ;
        halfD2Chi2DX2[5] += weight * trk.halfD2Chi2DX2()[5] ;

        halfDChi2DX.x   += weight * trk.halfDChi2DX().x ;
        halfDChi2DX.y   += weight * trk.halfDChi2DX().y ;
        halfDChi2DX.z   += weight * trk.halfDChi2DX().z ;

        chi2 += weight * trk.chi2() ;
      }
    }

    // check nr of tracks that entered the fit
    if(ntrin < m_minTr) {
      std::cout << "Too few tracks after PV fit" << std::endl;
      return false;
    }

    // compute the new vertex covariance
    for(int i = 0; i < 6; i++) vtxcov[i] = halfD2Chi2DX2[i] ;
    ROOT::Math::CholeskyDecomp<float, 3> decomp(vtxcov);
    if( !decomp ) return false;
    decomp.Invert(vtxcov);

    

    // compute the delta w.r.t. the reference
    XYZPoint delta{0.,0.,0.};
    delta.x = -1.0 * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z );
    delta.y = -1.0 * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z );
    delta.z = -1.0 * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z );

    // note: this is only correct if chi2 was chi2 of reference!
    chi2  += delta.x * halfDChi2DX.x + delta.y * halfDChi2DX.y + delta.z * halfDChi2DX.z;

    // deltaz needed for convergence
    const double deltaz = refpos.z + delta.z - vtxpos.z ;

    // update the position
    vtxpos.x = ( refpos.x + delta.x ) ;
    vtxpos.y = ( refpos.y + delta.y ) ;
    vtxpos.z = ( refpos.z + delta.z ) ;
    vtx.setChi2AndDoF( chi2, 2*ntrin-3 ) ;

    // loose convergence criteria if close to end of iterations
    if ( 1.*nbIter > 0.8*m_Iterations ) maxdz = 10.*m_maxDeltaZ;
    converged = std::abs(deltaz) < maxdz ;

  } // end iteration loop
  if(!converged) return false;

  
  // set position and covariance
  vtx.setPosition( vtxpos ) ;
  vtx.setCovMatrix( vtxcov ) ;
  // Set tracks. Compute final chi2.
  vtx.clearTracks();
  int tracks2remove_counter = 0;
  for( const auto& trk : pvTracks ) {
    if( trk.weight() > m_minTrackWeight)
      vtx.addToTracks( trk.track(), trk.weight() ) ;
    // remove track for next PV search
    if( trk.chi2(vtxpos) < m_trackMaxChi2Remove) {
      //tracks2remove[tracks2remove_counter] = trk.track() ;
      tracks2remove.push_back( trk.track() );
      tracks2remove_counter++;
    }
  }
  

  
  return true;
}

//=============================================================================
// Get Tukey's weight
//=============================================================================
double AdaptivePV3DFitter::getTukeyWeight(double trchi2, int iter) const
{
  if (iter<1 ) return 1.;
  double ctrv = m_trackChi * std::max(m_minIter -  iter,1);
  double cT2 = trchi2 / std::pow(ctrv*m_TrackErrorScaleFactor,2);
  return cT2 < 1. ? std::pow(1.-cT2,2) : 0. ;
}




