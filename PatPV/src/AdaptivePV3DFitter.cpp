#include <vector>


struct XYZPoint {
  double x = 0.;
  double y = 0.;
  double z = 0.;
  XYZPoint(double m_x, double m_y, double m_z) : x(m_x), y(m_y), z(m_z) {}

};

struct State {
  double tx = 0.;
  double ty = 0.;
  double x = 0.;
  double y = 0.;
  double z = 0.;
  double errX2 = 0.;
  double errY2 = 0.;


};

//typedef std::vector<State> Track;
class Track {
public:
  std::vector<State> states;
  State firstState() {
    return states.at(0);
  }
  XYZPoint slopes() {
    return XYZPoint(states.at(0).tx, states.at(0).ty, 1.);
  }
  XYZPoint position() {
    return XYZPoint(states.at(0).x, states.at(0).y, states.at(0).z);
  }

};

struct RecVertex {


};

class AdaptivePV3DFitter  {

public:
  // Standard constructor
  AdaptivePV3DFitter(){};
  // Fitting
  bool fitVertex(const XYZPoint& seedPoint,
                       const std::vector<const Track*>& tracks,
                       XYZPoint& vtx,
                       std::vector<const Track*>& tracks2remove) const;
private:
  size_t m_minTr = 4;
  int    m_Iterations = 20;
  int    m_minIter = 5;
  double m_maxDeltaZ = 0.0005; // unit:: mm
  double m_minTrackWeight = 0.00000001;
  double m_TrackErrorScaleFactor = 1.0;
  double m_maxChi2 = 400.0;
  double m_trackMaxChi2 = 12.;
  double m_trackChi ;     // sqrt of trackMaxChi2
  double m_trackMaxChi2Remove = 25.;
  double m_maxDeltaZCache = 1.; //unit: mm


  // Get Tukey's weight
  double getTukeyWeight(double trchi2, int iter) const;
};

struct Vector2 {
  double x;
  double y;

  Vector2(double m_x, double m_y) : x(m_x), y(m_y){}
};





  class AdaptivePVTrack 
  {
  public:
    AdaptivePVTrack( Track& track) ;
    void updateCache( const XYZPoint& vtx ) ;
    double weight() const { return m_weight ; }
    void setWeight(double w) { m_weight = w ;}
    const double *  halfD2Chi2DX2() const { return m_halfD2Chi2DX2 ; }
    const XYZPoint&  halfDChi2DX() const { return m_halfDChi2DX ; }
    double chi2() const { return m_chi2 ; }
    inline double chi2( const XYZPoint& vtx ) const ;
    const Track* track() const { return m_track ; }
  private:
    double m_weight ;
     Track* m_track ;
    State m_state ;
    //express symmetrical amtrices as arrays in in packed representation element m(i,j) (j <= i) is supposed to be in array element  (i * (i + 1)) / 2 + j


    double m_invcov[3] ;
    double m_halfD2Chi2DX2[6] ;
    XYZPoint m_halfDChi2DX{1.,2.,3.}  ;
    double m_chi2 ;
    double m_H[6];
  } ;


  AdaptivePVTrack::AdaptivePVTrack(Track& track)
    : m_track(&track)
  {
    // get the state
    m_state = track.firstState() ;

    // do here things we could evaluate at z_seed. may add cov matrix here, which'd save a lot of time.
    m_H[0] = 1 ;
    m_H[(2 * (2 + 1)) / 2 + 0] = - m_state.tx ;
    m_H[(2 * (2 + 1)) / 2 + 1] = - m_state.ty ;
    // update the cache
    //updateCache( vtx ) ;
  }

/*
  void AdaptivePVTrack::updateCache(const XYZPoint& vtx)
  {
    // transport to vtx z
    m_state.linearTransportTo( vtx.z() ) ;

    // invert cov matrix
    m_invcov = m_state.covariance().Sub<SymMatrix2x2>(0,0) ;
    m_invcov.InvertChol() ;

    // The following can all be written out, omitting the zeros, once
    // we know that it works.
    Vector2 res{ vtx.x() - m_state.x(), vtx.y() - m_state.y() };
    ROOT::Math::SMatrix<double,3,2> HW = m_H*m_invcov ;
    ROOT::Math::AssignSym::Evaluate(m_halfD2Chi2DX2, HW*ROOT::Math::Transpose(m_H) ) ;
    //m_halfD2Chi2DX2 = ROOT::Math::Similarity(H, invcov ) ;
    m_halfDChi2DX   = HW * res ;
    m_chi2          = ROOT::Math::Similarity(res,m_invcov) ;
  }

  inline double AdaptivePVTrack::chi2( const XYZPoint& vtx ) const
  {
    double dz = vtx.z() - m_state.z() ;
    Vector2 res{ vtx.x() - (m_state.x() + dz*m_state.tx()),
                        vtx.y() - (m_state.y() + dz*m_state.ty()) };
    return ROOT::Math::Similarity(res,m_invcov) ;
  }


//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
AdaptivePV3DFitter::AdaptivePV3DFitter()
  
{
  m_trackChi = std::sqrt(m_trackMaxChi2);
}

/*

//=============================================================================
// Least square adaptive fitting method
//=============================================================================
bool AdaptivePV3DFitter::fitVertex(const XYZPoint& seedPoint,
             const std::vector<const Track*>& rTracks,
             RecVertex& vtx,
             std::vector<const Track*>& tracks2remove) const
{
  tracks2remove.clear();

  // position at which derivatives are evaluated
  XYZPoint refpos = seedPoint ;

  // prepare tracks
  std::vector<AdaptivePVTrack> pvTracks ;
  pvTracks.reserve( rTracks.size() ) ;
  for( const auto& track : rTracks )
    if( track->hasVelo() ) {
      pvTracks.emplace_back( *track, refpos );
      if (pvTracks.back().chi2() >= m_maxChi2) pvTracks.pop_back();
    }

  if( pvTracks.size() < m_minTr ) {
    if(msgLevel(MSG::DEBUG)) debug() << "Too few tracks to fit PV" << endmsg;
    return false;
  }

  // current vertex position
  XYZPoint vtxpos = refpos ;
  // vertex covariance matrix
  SymMatrix3x3 vtxcov ;
  bool converged = false;
  double maxdz = m_maxDeltaZ;
  int nbIter = 0;
  while( (nbIter < m_minIter) || (!converged && nbIter < m_Iterations) )
  {
    ++nbIter;

    SymMatrix3x3 halfD2Chi2DX2 ;
    Vector3 halfDChi2DX ;
    // update cache if too far from reference position. this is the slow part.
    if( std::abs(refpos.z() - vtxpos.z()) > m_maxDeltaZCache ) {
      refpos = vtxpos ;
      for( auto& trk : pvTracks ) trk.updateCache( refpos ) ;
    }

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
        halfD2Chi2DX2 += weight * trk.halfD2Chi2DX2() ;
        halfDChi2DX   += weight * trk.halfDChi2DX() ;
        chi2 += weight * trk.chi2() ;
      }
    }

    // check nr of tracks that entered the fit
    if(ntrin < m_minTr) {
      if(msgLevel(MSG::DEBUG)) debug() << "Too few tracks after PV fit" << endmsg;
      return false;
    }

    // compute the new vertex covariance
    vtxcov = halfD2Chi2DX2 ;
    if (!vtxcov.InvertChol()) {
      if(msgLevel(MSG::DEBUG)) debug() << "Error inverting hessian matrix" << endmsg;
      return false;
    }
    // compute the delta w.r.t. the reference
    Vector3 delta = -1.0 * vtxcov * halfDChi2DX ;

    // note: this is only correct if chi2 was chi2 of reference!
    chi2  += ROOT::Math::Dot(delta,halfDChi2DX) ;

    // deltaz needed for convergence
    const double deltaz = refpos.z() + delta(2) - vtxpos.z() ;

    // update the position
    vtxpos.SetX( refpos.x() + delta(0) ) ;
    vtxpos.SetY( refpos.y() + delta(1) ) ;
    vtxpos.SetZ( refpos.z() + delta(2) ) ;
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
  for( const auto& trk : pvTracks ) {
    if( trk.weight() > m_minTrackWeight)
      vtx.addToTracks( trk.track(), trk.weight() ) ;
    // remove track for next PV search
    if( trk.chi2(vtxpos) < m_trackMaxChi2Remove)
      tracks2remove.push_back( trk.track() );
  }
  vtx.setTechnique(RecVertex::RecVertexType::Primary);
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

*/