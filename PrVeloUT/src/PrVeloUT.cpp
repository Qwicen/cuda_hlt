// Include files
// From LHCb - Kernel/LHCbKernel/Kernel/
#include "include/STLExtensions.h"

// local
#include "PrVeloUT.h"
//-----------------------------------------------------------------------------
// Implementation file for class : PrVeloUT
//
// 2007-05-08 : Mariusz Witek
// 2017-03-01: Christoph Hasse (adapt to future framework)
//-----------------------------------------------------------------------------

namespace {
  bool rejectTrack(const LHCb::Track* track){
    return track->checkFlag( LHCb::Track::Backward )
      || track->checkFlag( LHCb::Track::Invalid );
  }


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


// Declaration of the Algorithm Factory
// DECLARE_COMPONENT( PrVeloUT )
//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
PrVeloUT::PrVeloUT(const std::string& name,
                   ISvcLocator* pSvcLocator) :
Transformer(name, pSvcLocator,
            KeyValue{"InputTracksName", LHCb::TrackLocation::Velo} ,
            KeyValue{"OutputTracksName", LHCb::TrackLocation::VeloTT}){
}


//=============================================================================
// Initialization
//=============================================================================
int PrVeloUT::initialize() {

  // auto sc = Transformer::initialize();
  // if (sc.isFailure()) return sc;  // error printed already by GaudiAlgorithm

  // TODO not used?
  // m_veloUTTool = tool<ITracksFromTrackR>("PrVeloUTTool", this );

  m_PrUTMagnetTool = tool<PrUTMagnetTool>( "PrUTMagnetTool","PrUTMagnetTool");

  // m_zMidUT is a position of normalization plane which should to be close to z middle of UT ( +- 5 cm ).
  // Cashed once in PrVeloUTTool at initialization. No need to update with small UT movement.
  m_zMidUT    = m_PrUTMagnetTool->zMidUT();
  //  zMidField and distToMomentum isproperly recalculated in PrUTMagnetTool when B field changes
  m_distToMomentum = m_PrUTMagnetTool->averageDist2mom();

  // if (m_doTiming) {
  //   m_timerTool = tool<ISequencerTimerTool>( "SequencerTimerTool" );
  //   m_timerTool->increaseIndent();
  //   m_veloUTTime = m_timerTool->addTimer( "Internal VeloUT Tracking" );
  //   m_timerTool->decreaseIndent();
  // }

  m_sigmaVeloSlope = 0.10*Gaudi::Units::mrad;
  m_invSigmaVeloSlope = 1.0/m_sigmaVeloSlope;
  m_zKink = 1780.0;


  return 1;
}

//=============================================================================
// Main execution
//=============================================================================
// #ifndef SMALL_OUTPUT
LHCb::Track PrVeloUT::operator()(const LHCb::Track& inputTracks) const {
// #endif
// #ifdef SMALL_OUTPUT
// PrVeloUTTracks PrVeloUT::operator()(const LHCb::Tracks& inputTracks) const {
// #endif

//   // TODO
//   if ( m_doTiming ) m_timerTool->start( m_veloUTTime );

// #ifndef SMALL_OUTPUT
  LHCb::Track outputTracks;
// #endif
// #ifdef SMALL_OUTPUT
//   PrVeloUTTracks outputTracks;
// #endif

  outputTracks.reserve(inputTracks.size());

  // counter("#seeds") += inputTracks.size();

  const UT::HitHandler* hh = m_HitHandler.get();
  std::array<std::array<HitRange::const_iterator,85>,4> iteratorsLayers;

  fillIterators(hh, iteratorsLayers);

  const std::vector<float> fudgeFactors = m_PrUTMagnetTool->returnDxLayTable();
  const std::vector<float> bdlTable     = m_PrUTMagnetTool->returnBdlTable();

  std::array<UT::Mut::Hits,4> hitsInLayers;
  for( auto& it : hitsInLayers ) it.reserve(8); // check this number!

  for(const LHCb::Track* veloTr : inputTracks) {

    if( rejectTrack( veloTr ) ) continue;

    MiniState trState;
    if( !getState(veloTr, trState, outputTracks)) continue;

    for( auto& it : hitsInLayers ) it.clear();
    if( !getHits(hitsInLayers, iteratorsLayers, hh, fudgeFactors, trState) ) continue;

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
  
  // if ( m_doTiming ) m_timerTool->stop( m_veloUTTime );
  return outputTracks;
}
//=============================================================================
// Get the state, do some cuts
//=============================================================================
// #ifndef SMALL_OUTPUT
bool PrVeloUT::getState(const LHCb::Track* iTr, MiniState& trState, LHCb::Tracks& outputTracks) const {
// #endif
// #ifdef SMALL_OUTPUT
// bool PrVeloUT::getState(const LHCb::Track* iTr, MiniState& trState, PrVeloUTTracks& outputTracks) const {
// #endif
  const LHCb::State* s = iTr->stateAt(LHCb::State::EndVelo);
  const LHCb::State& state = s ? *s : (iTr->closestState(LHCb::State::EndVelo));

  // -- reject tracks outside of acceptance or pointing to the beam pipe
  trState.tx = state.tx();
  trState.ty = state.ty();
  trState.x = state.x();
  trState.y = state.y();
  trState.z = state.z();

  const float xMidUT =  trState.x + trState.tx*(m_zMidUT-trState.z);
  const float yMidUT =  trState.y + trState.ty*(m_zMidUT-trState.z);

  if( xMidUT*xMidUT+yMidUT*yMidUT  < m_centralHoleSize*m_centralHoleSize ) return false;
  if( (std::abs(trState.tx) > m_maxXSlope) || (std::abs(trState.ty) > m_maxYSlope) ) return false;

#ifndef SMALL_OUTPUT
  if(m_passTracks && std::abs(xMidUT) < m_passHoleSize && std::abs(yMidUT) < m_passHoleSize){
    std::unique_ptr<LHCb::Track> outTr{ new LHCb::Track() };
    outTr->reset();
    outTr->copy(*iTr);
    outTr->addToAncestors( iTr );
    outputTracks.insert(outTr.release());
    return false;
  }
#endif
#ifdef SMALL_OUTPUT
  if(m_passTracks && std::abs(xMidUT) < m_passHoleSize && std::abs(yMidUT) < m_passHoleSize){
    outputTracks.emplace_back();
    outputTracks.back().qOverP = 0;
    outputTracks.back().veloTr = iTr;

    return false;
  }
#endif

  return true;

}
//=============================================================================
// Find the hits
//=============================================================================
 bool PrVeloUT::getHits(std::array<UT::Mut::Hits,4>& hitsInLayers, const std::array<std::array<HitRange::const_iterator,85>,4>& iteratorsLayers,
                        const UT::HitHandler* hh,
                        const std::vector<float>& fudgeFactors, MiniState& trState ) const {

  // -- This is hardcoded, so faster
  // -- If you ever change the Table in the magnet tool, this will be wrong
  const float absSlopeY = std::abs( trState.ty );
  const int index = (int)(absSlopeY*100 + 0.5);
  const std::array<float,4> normFact = { fudgeFactors[4*index], fudgeFactors[1 + 4*index], fudgeFactors[2 + 4*index], fudgeFactors[3 + 4*index] };

  // -- this 500 seems a little odd...
  const float invTheta = std::min(500.,1.0/std::sqrt(trState.tx*trState.tx+trState.ty*trState.ty));
  const float minMom   = std::max(m_minPT.value()*invTheta, m_minMomentum.value());
  const float xTol     = std::abs(1. / ( m_distToMomentum * minMom ));
  const float yTol     = m_yTol + m_yTolSlope * xTol;

  int nLayers = 0;

  for(int iStation = 0; iStation < 2; ++iStation){

    if( iStation == 1 && nLayers == 0 ){
      //counter("#NoHitsFound")++;
      return false;
    }

    for(int iLayer = 0; iLayer < 2; ++iLayer){

      if( iStation == 1 && iLayer == 1 && nLayers < 2 ) return false;

      const HitRange& hits = hh->hits( iStation, iLayer );

      if( UNLIKELY( hits.empty() ) ) continue;

      const float dxDy   = hits.front().dxDy();
      const float zLayer = hits.front().zAtYEq0();

      const float yAtZ   = trState.y + trState.ty*(zLayer - trState.z);
      const float xLayer = trState.x + trState.tx*(zLayer - trState.z);
      const float yLayer = yAtZ + yTol*dxDyHelper[2*iStation+iLayer];

      const float normFactNum = normFact[2*iStation + iLayer];
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
      const float lowerBoundX =
        (xLayer - dxDy*yLayer) - xTol*invNormFact - std::abs(trState.tx)*m_intraLayerDist;
      const float upperBoundX =
        (xLayer - dxDy*yLayer) + xTol*invNormFact + std::abs(trState.tx)*m_intraLayerDist;

      const int indexLowProto = lowerBoundX > 0 ? std::sqrt( std::abs(lowerBoundX)*2.0 ) + 42 : 42 - std::sqrt( std::abs(lowerBoundX)*2.0 );
      const int indexHiProto  = upperBoundX > 0 ? std::sqrt( std::abs(upperBoundX)*2.0 ) + 43 : 43 - std::sqrt( std::abs(upperBoundX)*2.0 );

      const int indexLow  = std::max( indexLowProto, 0 );
      const int indexHi   = std::min( indexHiProto, 84);

      HitRange::const_iterator itBeg = iteratorsLayers[2*iStation+iLayer][ indexLow ];
      HitRange::const_iterator itEnd = iteratorsLayers[2*iStation+iLayer][ indexHi  ];

      while( (*itBeg).xAtYEq0() < lowerBoundX && itBeg != itEnd) ++itBeg;
      if(itBeg == itEnd) continue;

      findHits(itBeg, itEnd, trState, xTol*invNormFact, invNormFact, hitsInLayers[2*iStation + iLayer]);

      nLayers += int(!hitsInLayers[2*iStation + iLayer].empty());
    }
  }

  return nLayers > 2;
}

//=========================================================================
// Form clusters
//=========================================================================
bool PrVeloUT::formClusters(const std::array<UT::Mut::Hits,4>& hitsInLayers, TrackHelper& helper) const {

  bool fourLayerSolution = false;

  for(const auto& hit0 : hitsInLayers[0]){

    const float xhitLayer0 = hit0.x;
    const float zhitLayer0 = hit0.z;

    // Loop over Second Layer
    for(const auto& hit2 : hitsInLayers[2]){

      const float xhitLayer2 = hit2.x;
      const float zhitLayer2 = hit2.z;

      const float tx = (xhitLayer2 - xhitLayer0)/(zhitLayer2 - zhitLayer0);

      if( std::abs(tx-helper.state.tx) > m_deltaTx2 ) continue;

      const UT::Mut::Hit* bestHit1 = nullptr;
      float hitTol = m_hitTol2;
      for( auto& hit1 : hitsInLayers[1]){

        const float xhitLayer1 = hit1.x;
        const float zhitLayer1 = hit1.z;

        const float xextrapLayer1 = xhitLayer0 + tx*(zhitLayer1-zhitLayer0);
        if(std::abs(xhitLayer1 - xextrapLayer1) < hitTol){
          hitTol = std::abs(xhitLayer1 - xextrapLayer1);
          bestHit1 = &hit1;
        }
      }

      if( fourLayerSolution && !bestHit1) continue;

      const UT::Mut::Hit* bestHit3 = nullptr;
      hitTol = m_hitTol2;
      for( auto& hit3 : hitsInLayers[3]){

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
        simpleFit( LHCb::make_array( &hit0, bestHit1, &hit2, bestHit3 ), helper);

        if(!fourLayerSolution && helper.bestHits[0]){
          fourLayerSolution = true;
        }
        continue;
      }

      // -- Nothing found in layer 3
      if( !fourLayerSolution && bestHit1 ){
        simpleFit( LHCb::make_array( &hit0, bestHit1, &hit2 ),  helper);
        continue;
      }
      // -- Noting found in layer 1
      if( !fourLayerSolution && bestHit3 ){
        simpleFit( LHCb::make_array( &hit0, bestHit3, &hit2 ), helper);
        continue;
      }
    }
  }

  return fourLayerSolution;
}
//=========================================================================
// Create the Velo-TU tracks
//=========================================================================
#ifndef SMALL_OUTPUT
void PrVeloUT::prepareOutputTrack(const LHCb::Track* veloTrack,
                                  const TrackHelper& helper,
                                  const std::array<UT::Mut::Hits,4>& hitsInLayers,
                                  LHCb::Tracks& outputTracks,
                                  const std::vector<float>& bdlTable) const {
#endif
#ifdef SMALL_OUTPUT
void PrVeloUT::prepareOutputTrack(const LHCb::Track* veloTrack,
                                  const TrackHelper& helper,
                                  const std::array<UT::Mut::Hits,4>& hitsInLayers,
                                  PrVeloUTTracks& outputTracks,
                                  const std::vector<float>& bdlTable) const {
#endif

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

  const float qpxz2p =-1*vdt::fast_isqrt(1.+helper.state.ty*helper.state.ty)/bdl*3.3356/Gaudi::Units::GeV;
  const float qop = (std::abs(bdl) < 1.e-8) ? 0.0 : helper.bestParams[0]*qpxz2p;

  // -- Don't make tracks that have grossly too low momentum
  // -- Beware of the momentum resolution!
  const float p  = 1.3*std::abs(1/qop);
  const float pt = p*std::sqrt(helper.state.tx*helper.state.tx + helper.state.ty*helper.state.ty);

  if( p < m_minMomentum || pt < m_minPT ) return;

  const float txUT = helper.bestParams[3];

#ifndef SMALL_OUTPUT

  const float xUT  = helper.bestParams[2];
  std::unique_ptr<LHCb::Track> outTr{ new LHCb::Track()};

  // reset the track
  outTr->reset();
  outTr->copy(*veloTrack);

  // Adding overlap hits
  for( const auto* hit : helper.bestHits){

    // -- only the last one can be a nullptr.
    if( !hit ) break;

    outTr->addToLhcbIDs( hit->HitPtr->lhcbID() );

    const float xhit = hit->x;
    const float zhit = hit->z;

    for( auto& ohit : hitsInLayers[hit->HitPtr->planeCode()]){
      const float zohit = ohit.z;
      if(zohit==zhit) continue;

      const float xohit = ohit.x;
      const float xextrap = xhit + txUT*(zhit-zohit);
      if( xohit-xextrap < -m_overlapTol) continue;
      if( xohit-xextrap > m_overlapTol) break;
      outTr->addToLhcbIDs( ohit.HitPtr->lhcbID() );
      // -- only one overlap hit
      //break;
    }
  }

  // set q/p in all of the existing states
  for(auto& state : outTr->states()) state->setQOverP(qop);

  const float yMidUT =  helper.state.y + helper.state.ty*(m_zMidUT-helper.state.z);

  //== Add a new state...
  LHCb::State temp;
  temp.setLocation( LHCb::State::AtTT );
  temp.setState( xUT,
                 yMidUT,
                 m_zMidUT,
                 txUT,
                 helper.state.ty,
                 qop );


  outTr->addToStates( std::move(temp) );
  outTr->setType( LHCb::Track::Upstream );
  outTr->setHistory( LHCb::Track::PatVeloTT );
  outTr->addToAncestors( veloTrack );
  outTr->setPatRecStatus( LHCb::Track::PatRecIDs );
  outTr->setChi2PerDoF( helper.bestParams[1]);

  outputTracks.insert( outTr.release() );

#endif
#ifdef SMALL_OUTPUT

  outputTracks.emplace_back();
  //outputTracks.back().UTIDs.reserve(8);

  // Adding overlap hits
  for( const auto* hit : helper.bestHits){

    // -- only the last one can be a nullptr.
    if( !hit ) break;

    outputTracks.back().UTIDs.push_back(hit->HitPtr->lhcbID());

    const float xhit = hit->x;
    const float zhit = hit->z;

    for( auto& ohit : hitsInLayers[hit->HitPtr->planeCode()]){
      const float zohit = ohit.z;
      if(zohit==zhit) continue;

      const float xohit = ohit.x;
      const float xextrap = xhit + txUT*(zhit-zohit);
      if( xohit-xextrap < -m_overlapTol) continue;
      if( xohit-xextrap > m_overlapTol) break;
      outputTracks.back().UTIDs.push_back(ohit.HitPtr->lhcbID());
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
  outputTracks.back().qOverP = qop;
  outputTracks.back().veloTr = veloTrack;


#endif

}


