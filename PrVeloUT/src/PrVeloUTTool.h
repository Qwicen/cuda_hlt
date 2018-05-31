#ifndef PRVELOUTTOOL_H
#define PRVELOUTTOOL_H 1

// Include files
// from Gaudi
#include "Event/Track.h"

// local
#include "PrUTMagnetTool.h"
#include "PrKernel/UTHitHandler.h"
#include "PrKernel/UTHitInfo.h"
#include "PrKernel/UTHit.h"
#include "TrackInterfaces/ITracksFromTrackR.h"
#include "GaudiKernel/AnyDataHandle.h"

#include "vdt/sqrt.h"

/// helper class for internals of PrVeloUTTool

struct PrVeloUTEventState {
  std::array<UT::Hits,8> m_hitsLayers;
  std::array<UT::Mut::Hits,4> m_allHits;
};

struct PrVeloUTTrackState {
  std::array<float,4> m_normFact;
  std::array<float,4> m_invNormFact;
  UT::Mut::Hits m_clusterCandidate;
  UT::Mut::Hits m_bestCandHits;
  std::array<float,4> m_bestParams;
  float m_xVelo;
  float m_yVelo;
  float m_zVelo;
  float m_txVelo;
  float m_tyVelo;
  float m_yAtMidUT;
  float m_yAt0;
  bool m_fourLayerSolution;
  float m_xMid;
  float m_wb;
  float m_invKinkVeloDist;
};

  /** @class PrVeloUTTool PrVeloUTTool.h
   *
   *  PrVeloUT tool
   *
   *  @author Mariusz Witek
   *  @date   2007-05-08
   *  @update for A-Team framework 2007-08-20 SHM
   *
   *  2017-03-01: Christoph Hasse (adapt to future framework)
   *
   */

class PrVeloUTTool : public extends<GaudiTool, ITracksFromTrackR> {
public:
  /// Standard constructor
  PrVeloUTTool( const std::string& type,
                const std::string& name,
                const IInterface* parent);

  StatusCode initialize ( ) override;

  /// Create an instance of a state
  virtual ranges::v3::any createState() const override;

  /** main reconstruction method.
   * when called n times for tracks of the same event, the eventState
   * parameter should be passed to all calls. It is internally used to
   * in order to optimize computations
   */
  StatusCode tracksFromTrack(const LHCb::Track & velotrack,
                             std::vector<LHCb::Track*>& out,
                             ranges::v3::any& eventStateAsAny) const override;

private:

  std::unique_ptr<LHCb::Track> getCandidate(const LHCb::Track& veloTrack,
                                            PrVeloUTEventState &eventState,
                                            PrVeloUTTrackState &toolState) const;

  bool findHits(PrVeloUTEventState &eventState,
                PrVeloUTTrackState &toolState) const;

  void clustering() const;

  void formClusters(PrVeloUTEventState &eventState,
                    PrVeloUTTrackState &toolState) const;

  std::unique_ptr<LHCb::Track> prepareOutputTrack(const LHCb::Track& veloTrack,
                                                  PrVeloUTEventState &eventState,
                                                  PrVeloUTTrackState &toolState) const;

  void printHit( const UT::Hit* hit, const std::string title = "") const {
    info() << "   " << title
           << format("Plane%3d z0 %8.2f x0 %8.2f zAtYEq0 %8.2f xAtYEq0 %8.2f dxDy %8.3f yMin %8.2f yMax %8.2f highThreshold %3d",
                     hit->planeCode(), hit->zAtYEq0() , hit->xAtYMid(), hit->zAtYEq0(), hit->xAtYEq0(), hit->dxDy(), hit->yMin(), hit->yMax(), hit->highThreshold());
    info()<<endmsg;
  }

  void printHit(const UT::Mut::Hit& hit, const std::string title = "") const {
    info() << "    " << title
         << format("Mutables: x:%8.2f and z:%8.2f",hit.x, hit.z);
    info() << endmsg;
    printHit(hit.HitPtr,"Called Through MutUTHit ");
  }

  template<std::size_t nHits>
  void simpleFit(PrVeloUTTrackState &trackState) const;

private:

  Gaudi::Property<float> m_minMomentum    {this, "minMomentum",      0*Gaudi::Units::GeV};
  Gaudi::Property<float> m_minPT          {this, "minPT",            0.1*Gaudi::Units::GeV};
  Gaudi::Property<float> m_maxPseudoChi2  {this, "maxPseudoChi2",    1280.};
  Gaudi::Property<float> m_yTol           {this, "YTolerance",       0.8  * Gaudi::Units::mm};
  Gaudi::Property<float> m_yTolSlope      {this, "YTolSlope",        0.2};
  Gaudi::Property<float> m_hitTol1        {this, "HitTol1",          6.0 * Gaudi::Units::mm};
  Gaudi::Property<float> m_hitTol2        {this, "HitTol2",          0.8 * Gaudi::Units::mm};
  Gaudi::Property<float> m_deltaTx1       {this, "DeltaTx1",         0.035};
  Gaudi::Property<float> m_deltaTx2       {this, "DeltaTx2",         0.02};
  Gaudi::Property<float> m_maxXSlope      {this, "MaxXSlope",        0.350};
  Gaudi::Property<float> m_maxYSlope      {this, "MaxYSlope",        0.300};
  Gaudi::Property<float> m_centralHoleSize{this, "centralHoleSize",  33. * Gaudi::Units::mm};
  Gaudi::Property<float> m_intraLayerDist {this, "IntraLayerDist",   15.0 * Gaudi::Units::mm};
  Gaudi::Property<float> m_overlapTol     {this, "OverlapTol",       0.7 * Gaudi::Units::mm};
  Gaudi::Property<float> m_passHoleSize   {this, "PassHoleSize",     40. * Gaudi::Units::mm};
  Gaudi::Property<int>   m_minHighThres   {this, "MinHighThreshold", 1};
  Gaudi::Property<bool>  m_printVariables {this, "PrintVariables",   false};
  Gaudi::Property<bool>  m_passTracks     {this, "PassTracks",       false};

  AnyDataHandle<UT::HitHandler> m_HitHandler {UT::Info::HitLocation, Gaudi::DataHandle::Reader, this};

  float m_zMidUT;
  float m_distToMomentum;
  float m_zKink;

  PrUTMagnetTool*    m_PrUTMagnetTool;  ///< Multipupose tool for Bdl and deflection

  float m_sigmaVeloSlope;
  float m_invSigmaVeloSlope;
};


//=========================================================================
// A kind of global track fit in VELO and TT
// The pseudo chi2 consists of two contributions:
//  - chi2 of Velo track x slope
//  - chi2 of a line in TT
// The two track segments go via the same (x,y) point
// at z corresponding to the half Bdl of the track
//
// Only q/p and chi2 of outTr are modified
//
//=========================================================================
template<std::size_t nHits>
void PrVeloUTTool::simpleFit(PrVeloUTTrackState &trackState) const {

  //this guy is a MuUTHit
  const auto& theHits = trackState.m_clusterCandidate;

  const float zDiff = 0.001*(m_zKink-m_zMidUT);
  float mat[3] = { trackState.m_wb, trackState.m_wb*zDiff, trackState.m_wb*zDiff*zDiff };
  float rhs[2] = { trackState.m_wb*trackState.m_xMid, trackState.m_wb*trackState.m_xMid*zDiff };

  // -- Scale the z-component, to not run into numerical problems
  // -- with floats
  int nHighThres = 0;
  for ( std::size_t i = 0; i < nHits; ++i){
    const auto& hit = theHits[i];
    if(  hit.HitPtr->highThreshold() ) ++nHighThres;
    const float ui = hit.x;
    const float ci = hit.HitPtr->cosT();
    const float dz = 0.001*(hit.z - m_zMidUT);
    const float wi = hit.HitPtr->weight();
    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }

  // -- Veto hit combinations with no high threshold hit
  // -- = likely spillover
  if( nHighThres < m_minHighThres ) return;


  ROOT::Math::CholeskyDecomp<float, 2> decomp(mat);
  if( UNLIKELY(!decomp)) {
    return;
  } else {
    decomp.Solve(rhs);
  }

  const float xSlopeTTFit = 0.001*rhs[1];
  const float xTTFit = rhs[0];

  // new VELO slope x
  const float xb = xTTFit+xSlopeTTFit*(m_zKink-m_zMidUT);
  const float xSlopeVeloFit = (xb-trackState.m_xVelo)*trackState.m_invKinkVeloDist;
  const float chi2VeloSlope = (trackState.m_txVelo - xSlopeVeloFit)*m_invSigmaVeloSlope;

  float chi2TT = 0;
  for ( std::size_t i = 0; i < nHits; ++i){

    //this guy is a MuUTHit
    const auto& hit = theHits[i];

    const float zd    = hit.z;
    const float xd    = xTTFit + xSlopeTTFit*(zd-m_zMidUT);
    const float du    = xd - hit.x;

    chi2TT += (du*du)*hit.HitPtr->weight();

  }

  chi2TT += chi2VeloSlope*chi2VeloSlope;
  chi2TT /= (nHits + 1 - 2);


  if( chi2TT < trackState.m_bestParams[1] ){

    // calculate q/p
    const float sinInX  = xSlopeVeloFit*vdt::fast_isqrt(1.+xSlopeVeloFit*xSlopeVeloFit);
    const float sinOutX = xSlopeTTFit*vdt::fast_isqrt(1.+xSlopeTTFit*xSlopeTTFit);
    const float qp = (sinInX-sinOutX);

    trackState.m_bestParams = { qp, chi2TT, xTTFit,xSlopeTTFit };
    trackState.m_bestCandHits = theHits;
  }

}

#endif // PATVELOTTTOOL_H
