#include "PrVeloUT.h"
//-----------------------------------------------------------------------------
// Implementation file for class : PrVeloUT
//
// 2007-05-08: Mariusz Witek
// 2017-03-01: Christoph Hasse (adapt to future framework)
// 2018-05-05: Plácido Fernández (make standalone)
//-----------------------------------------------------------------------------

namespace {
  // bool rejectTrack(const Track* track){
  //   return track->checkFlag( Track::Backward )
  //     || track->checkFlag( Track::Invalid );
  // }


  // -- These things are all hardcopied from the PrTableForFunction
  // -- and PrUTMagnetTool
  // -- If the granularity or whatever changes, this will give wrong results

  int masterIndex(const int index1, const int index2, const int index3){
    return (index3*11 + index2)*31 + index1;
  }

  constexpr std::array<float,3> minValsBdl = { -0.3, -250.0, 0.0 };
  constexpr std::array<float,3> maxValsBdl = { 0.3, 250.0, 800.0 };
  constexpr std::array<float,3> deltaBdl   = { 0.02, 50.0, 80.0 };

  constexpr std::array<float,4> dxDyHelper = { 0.0, 1.0, -1.0, 0.0 };
}

//=============================================================================
// Initialization
//=============================================================================
int PrVeloUT::initialize() {

  // TODO not used? old version?
  // m_veloUTTool = tool<ITracksFromTrackR>("PrVeloUTTool", this );

  // m_PrUTMagnetTool = PrUTMagnetTool(filenames);

  // m_zMidUT is a position of normalization plane which should to be close to z middle of UT ( +- 5 cm ).
  // Cashed once in PrVeloUTTool at initialization. No need to update with small UT movement.
  m_zMidUT    = m_PrUTMagnetTool.zMidUT();
  //  zMidField and distToMomentum isproperly recalculated in PrUTMagnetTool when B field changes
  m_distToMomentum = m_PrUTMagnetTool.averageDist2mom();

  m_sigmaVeloSlope = 0.10*Gaudi::Units::mrad;
  m_invSigmaVeloSlope = 1.0/m_sigmaVeloSlope;
  m_zKink = 1780.0;

  return 1;
}

//=============================================================================
// Main execution
//=============================================================================
std::vector<VeloUTTracking::TrackVelo> PrVeloUT::operator() (
  const std::vector<VeloUTTracking::TrackVelo>& inputTracks) const 
{

  std::vector<VeloUTTracking::TrackVelo> outputTracks;
  outputTracks.reserve(inputTracks.size());

  const std::vector<float> fudgeFactors = m_PrUTMagnetTool.returnDxLayTable();
  const std::vector<float> bdlTable     = m_PrUTMagnetTool.returnBdlTable();

  std::array<std::vector<VeloUTTracking::Hit>,4> hitsInLayers;
  // TODO get the proper hits
  const std::array<std::vector<VeloUTTracking::Hit>,4> inputHits;

  for(const VeloUTTracking::TrackVelo& veloTr : inputTracks) {

    VeloState trState;
    if( !getState(veloTr, trState, outputTracks)) continue;
    if( !getHits(hitsInLayers, inputHits, fudgeFactors, trState) ) continue;

    TrackHelper helper(trState, m_zKink, m_sigmaVeloSlope, m_maxPseudoChi2);

    //counter("formingClusters")++;
    if( !formClusters(hitsInLayers, helper) ){
      std::reverse(hitsInLayers.begin(),hitsInLayers.end());
      //counter("reversing")++;
      formClusters(hitsInLayers, helper);
      std::reverse(hitsInLayers.begin(),hitsInLayers.end());
    }

    if( helper.bestHits[0]){
      prepareOutputTrack(veloTr, helper, hitsInLayers, outputTracks, bdlTable);
    }

  }

  // counter("#tracks") += outputTracks.size();

  return outputTracks;
}

//=============================================================================
// Get the state, do some cuts
//=============================================================================
bool PrVeloUT::getState(
  const VeloUTTracking::TrackVelo& iTr, 
  VeloState& trState, 
  std::vector<VeloUTTracking::TrackVelo>& outputTracks ) const 
{
  // const VeloState* s = iTr.stateAt(LHCb::State::EndVelo);
  // const VeloState& state = s ? *s : (iTr.closestState(LHCb::State::EndVelo));
  // TODO get the closest state not the last
  const VeloState state = iTr.back();

  // -- reject tracks outside of acceptance or pointing to the beam pipe
  trState.tx = state.tx;
  trState.ty = state.ty;
  trState.x = state.x;
  trState.y = state.y;
  trState.z = state.z;

  // m_zMidUT comes from MagnetTool
  const float xMidUT =  trState.x + trState.tx*( m_zMidUT - trState.z);
  const float yMidUT =  trState.y + trState.ty*( m_zMidUT - trState.z);

  if( xMidUT*xMidUT+yMidUT*yMidUT  < m_centralHoleSize*m_centralHoleSize ) return false;
  if( (std::abs(trState.tx) > m_maxXSlope) || (std::abs(trState.ty) > m_maxYSlope) ) return false;

  if(m_passTracks && std::abs(xMidUT) < m_passHoleSize && std::abs(yMidUT) < m_passHoleSize) {

    // TODO confirm this
    // std::unique_ptr<LHCb::Track> outTr{ new LHCb::Track() };
    // outTr->reset();
    // outTr->copy(*iTr);
    // outTr->addToAncestors( iTr );
    // outputTracks.insert(outTr.release());

    outputTracks.emplace_back(iTr);  // DvB: is this to save Velo tracks that don't make it to the UT?

    return false;
  }

  return true;

}

//=============================================================================
// Find the hits
//=============================================================================
bool PrVeloUT::getHits(
  std::array<std::vector<VeloUTTracking::Hit>,4>& hitsInLayers,
  const std::array<std::vector<VeloUTTracking::Hit>,4>& inputHits,
  const std::vector<float>& fudgeFactors, 
  const VeloState& trState ) const 
{
  // -- This is hardcoded, so faster
  // -- If you ever change the Table in the magnet tool, this will be wrong
  const float absSlopeY = std::abs( trState.ty );
  const int index = (int)(absSlopeY*100 + 0.5);
  const std::array<float,4> normFact = { 
    fudgeFactors[4*index], 
    fudgeFactors[1 + 4*index], 
    fudgeFactors[2 + 4*index], 
    fudgeFactors[3 + 4*index] 
  };

  // -- this 500 seems a little odd...
  const float invTheta = std::min(500.,1.0/std::sqrt(trState.tx*trState.tx+trState.ty*trState.ty));
  const float minMom   = std::max(m_minPT*invTheta, m_minMomentum);
  const float xTol     = std::abs(1. / ( m_distToMomentum * minMom ));
  const float yTol     = m_yTol + m_yTolSlope * xTol;

  int nLayers = 0;

  for(int iStation = 0; iStation < 2; ++iStation) {

    if( iStation == 1 && nLayers == 0 ) return false;

    for(int iLayer = 0; iLayer < 2; ++iLayer) {
      if( iStation == 1 && iLayer == 1 && nLayers < 2 ) return false;

      int layer = 2*iStation+iLayer;
      const std::vector<VeloUTTracking::Hit>& hits = inputHits[layer];

      // UNLIKELY is a MACRO for `__builtin_expect` compiler hint of GCC
      if( hits.empty() ) continue;

      const float dxDy   = hits.front().dxDy();
      const float zLayer = hits.front().zAtYEq0();

      const float yAtZ   = trState.y + trState.ty*(zLayer - trState.z);
      const float xLayer = trState.x + trState.tx*(zLayer - trState.z);
      const float yLayer = yAtZ + yTol*dxDyHelper[layer];

      const float normFactNum = normFact[layer];
      const float invNormFact = 1.0/normFactNum;

      // -- Get the (approximate) x position at y=0. For stereo layers, we need to take
      // -- dx/dy into account and shift along a strip.
      // -- last term is to take different z positions of modules into account
      // -- max distance between strips in a layer => 15mm
      // -- Hits are sorted at y=0
      //
      // -- It turns out that putting iterators at certain positions
      // -- and then do a linear search is faster than binary
      // -- searching the whole range (or binary searching)
      // -- starting from the iterator positions.

      // const float lowerBoundX =
      //   (xLayer - dxDy*yLayer) - xTol*invNormFact - std::abs(trState.tx)*m_intraLayerDist;
      // const float upperBoundX =
      //   (xLayer - dxDy*yLayer) + xTol*invNormFact + std::abs(trState.tx)*m_intraLayerDist;

      // const int indexLowProto = lowerBoundX > 0 ? std::sqrt( std::abs(lowerBoundX)*2.0 ) + 42 : 42 - std::sqrt( std::abs(lowerBoundX)*2.0 );
      // const int indexHiProto  = upperBoundX > 0 ? std::sqrt( std::abs(upperBoundX)*2.0 ) + 43 : 43 - std::sqrt( std::abs(upperBoundX)*2.0 );

      // const int indexLow  = std::max( indexLowProto, 0 );
      // const int indexHi   = std::min( indexHiProto, 84);

      // HitRange::const_iterator itBeg = iteratorsLayers[layer][ indexLow ];
      // HitRange::const_iterator itEnd = iteratorsLayers[layer][ indexHi  ];

      // while( (*itBeg).xAtYEq0() < lowerBoundX && itBeg != itEnd) ++itBeg;
      // if(itBeg == itEnd) continue;

      findHits(hits, trState, xTol*invNormFact, invNormFact, hitsInLayers[layer]);

      nLayers += int(!hitsInLayers[layer].empty());
    }
  }

  return nLayers > 2;
}

//=========================================================================
// Form clusters
//=========================================================================
bool PrVeloUT::formClusters(
  const std::array<std::vector<VeloUTTracking::Hit>,4>& hitsInLayers, 
  TrackHelper& helper ) const 
{

  bool fourLayerSolution = false;

  for(const auto& hit0 : hitsInLayers[0]) {

    const float xhitLayer0 = hit0.x;
    const float zhitLayer0 = hit0.z;

    // Loop over Second Layer
    for(const auto& hit2 : hitsInLayers[2]) {

      const float xhitLayer2 = hit2.x;
      const float zhitLayer2 = hit2.z;

      const float tx = (xhitLayer2 - xhitLayer0)/(zhitLayer2 - zhitLayer0);

      if( std::abs(tx-helper.state.tx) > m_deltaTx2 ) continue;

      const VeloUTTracking::Hit* bestHit1 = nullptr;
      float hitTol = m_hitTol2;
      for(const auto& hit1 : hitsInLayers[1]) {

        const float xhitLayer1 = hit1.x;
        const float zhitLayer1 = hit1.z;

        const float xextrapLayer1 = xhitLayer0 + tx*(zhitLayer1-zhitLayer0);
        if(std::abs(xhitLayer1 - xextrapLayer1) < hitTol){
          hitTol = std::abs(xhitLayer1 - xextrapLayer1);
          bestHit1 = &hit1;
        }
      }

      if( fourLayerSolution && !bestHit1) continue;

      const VeloUTTracking::Hit* bestHit3 = nullptr;
      hitTol = m_hitTol2;
      for( auto& hit3 : hitsInLayers[3]) {

        const float xhitLayer3 = hit3.x;
        const float zhitLayer3 = hit3.z;

        const float xextrapLayer3 = xhitLayer2 + tx*(zhitLayer3-zhitLayer2);

        if(std::abs(xhitLayer3 - xextrapLayer3) < hitTol){
          hitTol = std::abs(xhitLayer3 - xextrapLayer3);
          bestHit3 = &hit3;
        }
      }

      // -- All hits found
      if( bestHit1 && bestHit3 ){
        std::array<const VeloUTTracking::Hit*,4> hits4fit = {&hit0, bestHit1, &hit2, bestHit3};
        simpleFit(hits4fit, helper);

        if(!fourLayerSolution && helper.bestHits[0]){
          fourLayerSolution = true;
        }
        continue;
      }

      // -- Nothing found in layer 3
      if( !fourLayerSolution && bestHit1 ){
        std::array<const VeloUTTracking::Hit*,3> hits4fit = {&hit0, bestHit1, &hit2};
        simpleFit(hits4fit,  helper);
        continue;
      }
      // -- Noting found in layer 1
      if( !fourLayerSolution && bestHit3 ){
        std::array<const VeloUTTracking::Hit*,3> hits4fit = {&hit0, bestHit3, &hit2};
        simpleFit(hits4fit, helper);
        continue;
      }
    }
  }

  return fourLayerSolution;
}
//=========================================================================
// Create the Velo-TU tracks
//=========================================================================
void PrVeloUT::prepareOutputTrack(const VeloUTTracking::TrackVelo& veloTrack,
                                  const TrackHelper& helper,
                                  const std::array<std::vector<VeloUTTracking::Hit>,4>& hitsInLayers,
                                  std::vector<VeloUTTracking::TrackVelo>& outputTracks,
                                  const std::vector<float>& bdlTable) const {

  //== Handle states. copy Velo one, add TT.
  const float zOrigin = (std::fabs(helper.state.ty) > 0.001)
    ? helper.state.z - helper.state.y / helper.state.ty
    : helper.state.z - helper.state.x / helper.state.tx;

  //const float bdl1    = m_PrUTMagnetTool->bdlIntegral(helper.state.ty,zOrigin,helper.state.z);

  // -- These are calculations, copied and simplified from PrTableForFunction
  const std::array<float,3> var = { helper.state.ty, zOrigin, helper.state.z };

  const int index1 = std::max(0, std::min( 30, int((var[0] + 0.3)/0.6*30) ));
  const int index2 = std::max(0, std::min( 10, int((var[1] + 250)/500*10) ));
  const int index3 = std::max(0, std::min( 10, int( var[2]/800*10)        ));

  float bdl = bdlTable[masterIndex(index1, index2, index3)];

  const std::array<float,3> bdls = { bdlTable[masterIndex(index1+1, index2,index3)],
                                     bdlTable[masterIndex(index1,index2+1,index3)],
                                     bdlTable[masterIndex(index1,index2,index3+1)] };

  const std::array<float,3> boundaries = { -0.3f + float(index1)*deltaBdl[0],
                                           -250.0f + float(index2)*deltaBdl[1],
                                           0.0f + float(index3)*deltaBdl[2] };

  // -- This is an interpolation, to get a bit more precision
  float addBdlVal = 0.0;
  for(int i=0; i<3; ++i) {

    if( var[i] < minValsBdl[i] || var[i] > maxValsBdl[i] ) continue;

    const float dTab_dVar =  (bdls[i] - bdl) / deltaBdl[i];
    const float dVar = (var[i]-boundaries[i]);
    addBdlVal += dTab_dVar*dVar;
  }
  bdl += addBdlVal;
  // ----

  const float qpxz2p =-1*std::sqrt(1.+helper.state.ty*helper.state.ty)/bdl*3.3356/Gaudi::Units::GeV;
  const float qop = (std::abs(bdl) < 1.e-8) ? 0.0 : helper.bestParams[0]*qpxz2p;

  // -- Don't make tracks that have grossly too low momentum
  // -- Beware of the momentum resolution!
  const float p  = 1.3*std::abs(1/qop);
  const float pt = p*std::sqrt(helper.state.tx*helper.state.tx + helper.state.ty*helper.state.ty);

  if( p < m_minMomentum || pt < m_minPT ) return;

  const float txUT = helper.bestParams[3];

  outputTracks.emplace_back();
  //outputTracks.back().UTIDs.reserve(8);

  // Adding overlap hits
  for( const auto* hit : helper.bestHits){

    // -- only the last one can be a nullptr.
    if( !hit ) break;

    // TODO add a TrackStructure with UTIDs
    // outputTracks.back().UTIDs.push_back(hit->lhcbID());

    const float xhit = hit->x;
    const float zhit = hit->z;

    for( auto& ohit : hitsInLayers[hit->planeCode()]){
      const float zohit = ohit.z;
      if(zohit==zhit) continue;

      const float xohit = ohit.x;
      const float xextrap = xhit + txUT*(zhit-zohit);
      if( xohit-xextrap < -m_overlapTol) continue;
      if( xohit-xextrap > m_overlapTol) break;
      // TODO add a TrackStructure with UTIDs
      // outputTracks.back().UTIDs.push_back(ohit.lhcbID());
      // -- only one overlap hit
      break;
    }
  }

  /*
  outTr.x = helper.state.x;
  outTr.y = helper.state.y;
  outTr.z = helper.state.z;
  outTr.tx = helper.state.tx;
  outTr.ty = helper.state.ty;
  */

  // TODO add this to a track structure
  // outputTracks.back().qOverP = qop;
  // outputTracks.back().veloTr = veloTrack;


// #endif

}


