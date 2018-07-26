#include <vector>
#include <cmath>
#include <iostream>

#include "../include/AdaptivePV3DFitter.h"
//#include "../../PrVeloUT/include/CholeskyDecomp.h"

  size_t m_minTr = 4;
  int    m_Iterations = 20;
  int    m_minIter = 5;
  double m_maxDeltaZ = 0.0005; // unit:: mm
  double m_minTrackWeight = 0.00000001;
  double m_TrackErrorScaleFactor = 1.0;
  double m_maxChi2 = 400.0;
  double m_trackMaxChi2 = 12.;
  double m_trackChi = std::sqrt(m_trackMaxChi2);     // sqrt of trackMaxChi2
  double m_trackMaxChi2Remove = 25.;
  double m_maxDeltaZCache = 1.; //unit: mm



//=============================================================================
// Least square adaptive fitting method
//=============================================================================
bool fitVertex( XYZPoint& seedPoint,
              VeloState * host_velo_states,
             Vertex& vtx,
             std::vector<VeloState>& tracks2remove, int number_of_tracks, bool * tracks2disable) 
{
  //tracks2remove.clear();

  // position at which derivatives are evaluated
  XYZPoint refpos = seedPoint ;

  // prepare tracks
  AdaptivePVTrack pvTracks[number_of_tracks] ;

  int pvTrack_counter = 0;
  //for( const auto& track : rTracks ) {
  for(int i = 0; i < number_of_tracks; i++) {  
      AdaptivePVTrack pvTrack(host_velo_states[i], refpos);

      //don't use disabled tracks
      if(tracks2disable[i]) continue;
      if(pvTrack.chi2() < m_maxChi2) {
        pvTracks[pvTrack_counter] = pvTrack;
        pvTrack_counter++;
      }

      
  }
    

  if( pvTrack_counter < m_minTr ) {
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
      for( int index = 0; index < pvTrack_counter; index++)  pvTracks[index].updateCache( refpos ) ;
    };

    // add contribution from all tracks
    double chi2(0) ;
    size_t ntrin(0) ;
    for( int index = 0; index < pvTrack_counter; index++) {
      // compute weight
      AdaptivePVTrack trk = pvTracks[index];
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


    //repalce Cholesky inverter by analytical solution
    double a00 = halfD2Chi2DX2[0];
    double a10 = halfD2Chi2DX2[1];
    double a11 = halfD2Chi2DX2[2];
    double a20 = halfD2Chi2DX2[3];
    double a21 = halfD2Chi2DX2[4];
    double a22 = halfD2Chi2DX2[5];

    double det = a00 * (a22 * a11 - a21 * a21) - a10 * (a22 * a10 - a21 * a20) + a20 * (a21*a10 - a11*a20);
    if (det == 0) return false;


   vtxcov[0] = (a22*a11 - a21*a21) / det;
   vtxcov[1] = -(a22*a10-a20*a21) / det;
   vtxcov[2] = (a22*a00-a20*a20) / det;
   vtxcov[3] = (a21*a10-a20*a11) / det;
   vtxcov[4] = -(a21*a00-a20*a10) / det;
   vtxcov[5] = (a11*a00-a10*a10) / det;

    

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
  for( int index = 0; index < pvTrack_counter; index++) {
    AdaptivePVTrack trk = pvTracks[index];
    if( trk.weight() > m_minTrackWeight)
      vtx.addToTracks( trk.track(), trk.weight() ) ;
    // remove track for next PV search
    if( trk.chi2(vtxpos) < m_trackMaxChi2Remove) {
      //tracks2remove[tracks2remove_counter] = trk.track() ;
      tracks2remove.push_back( trk.track() );
      tracks2remove_counter++;
    }
  }


  //disable tracks added to this vertex
    for(int i = 0; i < number_of_tracks; i++) {  
      AdaptivePVTrack pvTrack(host_velo_states[i], refpos);

      //don't use disabled tracks
      if(tracks2disable[i]) continue;
      if( pvTrack.chi2(vtxpos) < m_trackMaxChi2Remove) tracks2disable[i] = true;
   
  }
  

  
  return true;
}

//=============================================================================
// Get Tukey's weight
//=============================================================================
double getTukeyWeight(double trchi2, int iter) 
{
  if (iter<1 ) return 1.;
  double ctrv = m_trackChi * std::max(m_minIter -  iter,1);
  double cT2 = trchi2 / std::pow(ctrv*m_TrackErrorScaleFactor,2);
  return cT2 < 1. ? std::pow(1.-cT2,2) : 0. ;
}




