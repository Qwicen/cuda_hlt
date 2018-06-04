// Include files

// from Gaudi
// #include "TfKernel/RecoFuncs.h"

// local
#include "PrVeloUTTool.h"

//-----------------------------------------------------------------------------
// Implementation file for class : PrVeloUTTool
//
// 2007-05-08 : Mariusz Witek
// 2017-03-01: Christoph Hasse (adapt to future framework)
//-----------------------------------------------------------------------------

namespace {
  struct  lowerBoundX  {
    bool operator() (const UT::Hit* first, const float value ) const {
      return first->xAtYMid() < value ;
    }
  };

  bool acceptTrack(const LHCb::Track& track){
    return !track.checkFlag( LHCb::Track::Invalid )
      && !track.checkFlag( LHCb::Track::Backward );
  }

}


// Declaration of the Tool Factory
DECLARE_COMPONENT( PrVeloUTTool )
//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
PrVeloUTTool::PrVeloUTTool( const std::string& type,
                            const std::string& name,
                            const IInterface* parent )
 : base_class( type, name , parent )
{
  declareInterface<ITracksFromTrackR>(this);
  declareProperty( "HitHandlerLoc", m_HitHandler ) ;
}

//=========================================================================
//  Initialisation, check parameters
//=========================================================================
StatusCode PrVeloUTTool::initialize ( ) {
  StatusCode sc = GaudiTool::initialize();
  if ( !sc ) return sc;

  m_PrUTMagnetTool = tool<PrUTMagnetTool>( "PrUTMagnetTool","PrUTMagnetTool");

  // m_zMidTT is a position of normalization plane which should to be close to z middle of TT ( +- 5 cm ).
  // Cashed once in PrVeloUTTool2 at initialization. No need to update with small TT movement.
  m_zMidUT    = m_PrUTMagnetTool->zMidUT();
  //  zMidField and distToMomentum isproperly recalculated in PrUTMagnetTool when B field changes
  m_distToMomentum = m_PrUTMagnetTool->averageDist2mom();

  if(m_printVariables){
    info() << " minMomentum        = " << m_minMomentum      << " MeV" << endmsg;
    info() << " minPT              = " << m_minPT            << " MeV" << endmsg;
    info() << " maxPseudoChi2      = " << m_maxPseudoChi2    << "   "  << endmsg;
    info() << " MaxXSlope          = " << m_maxXSlope    << "   "  << endmsg;
    info() << " MaxYSlope          = " << m_maxYSlope    << "   "  << endmsg;
    info() << " distToMomentum     = " << m_distToMomentum               << endmsg;
    info() << " yTolerance         = " << m_yTol             << " mm"  << endmsg;
    info() << " YTolSlope          = " << m_yTolSlope                  << endmsg;
    info() << " HitTol1            = " << m_hitTol1       << " mm " << endmsg;
    info() << " HitTol2            = " << m_hitTol2       << " mm " << endmsg;
    info() << " DeltaTx1           = " << m_deltaTx1       << "  " << endmsg;
    info() << " DeltaTx2           = " << m_deltaTx2       << "  " << endmsg;
    info() << " zMidUT             = " << m_zMidUT           << " mm"  << endmsg;
    info() << " IntraLayerDist     = " << m_intraLayerDist << " mm "<<endmsg;
    info() << " PassTracks         = " <<m_passTracks <<endmsg;
    info() << " PassHoleSize       = " <<m_passHoleSize << " mm "<<endmsg;
    info() << " OverlapTol         = " << m_overlapTol       << " mm " << endmsg;
  }

  m_sigmaVeloSlope = 0.10*Gaudi::Units::mrad;
  m_invSigmaVeloSlope = 1.0/m_sigmaVeloSlope;
  m_zKink = 1780.0;

  return StatusCode::SUCCESS;
}

//=========================================================================
// Main reconstruction method
//=========================================================================
StatusCode PrVeloUTTool::tracksFromTrack(const LHCb::Track & velotrack,
                                         std::vector<LHCb::Track*>& out,
                                         ranges::v3::any& eventStateAsAny) const
{
  PrVeloUTEventState &eventState = ranges::v3::any_cast<PrVeloUTEventState&>(eventStateAsAny);
  PrVeloUTTrackState trackState;

  if( msgLevel(MSG::DEBUG) ){
    debug()<<"RecoVeloUT method"<<endmsg;
  }

  //Remove backward/invalid tracks
   if(!acceptTrack(velotrack)){ return StatusCode::SUCCESS; }

  //Save some variables
  const LHCb::State& state = velotrack.hasStateAt(LHCb::State::LastMeasurement) ?
    *(velotrack.stateAt(LHCb::State::LastMeasurement)) :
    (velotrack.closestState(LHCb::State::EndVelo)) ;

  trackState.m_xVelo = state.x();
  trackState.m_yVelo = state.y();
  trackState.m_zVelo = state.z();
  trackState.m_txVelo = state.tx();
  trackState.m_tyVelo = state.ty();

  //Skip tracks outside
  if((fabs(trackState.m_txVelo) > m_maxXSlope) || (fabs(trackState.m_tyVelo) > m_maxYSlope)) return StatusCode::SUCCESS;

  float xAtMidTT = trackState.m_xVelo + trackState.m_txVelo*(m_zMidUT-trackState.m_zVelo);
  trackState.m_yAtMidUT = trackState.m_yVelo + trackState.m_tyVelo*(m_zMidUT-trackState.m_zVelo);

  // skip tracks pointing into central hole of TT
  if(sqrt(xAtMidTT*xAtMidTT+trackState.m_yAtMidUT*trackState.m_yAtMidUT) < m_centralHoleSize) return StatusCode::SUCCESS;

  if(m_passTracks && fabs(xAtMidTT)< m_passHoleSize && fabs(trackState.m_yAtMidUT) < m_passHoleSize){
    std::unique_ptr<LHCb::Track> outTr{ new LHCb::Track() };
    outTr->reset();
    outTr->copy(velotrack);
    outTr->addToAncestors( velotrack );

    out.push_back(outTr.release());
    return StatusCode::SUCCESS;
  }

  //clear vectors
  trackState.m_normFact = { 1.0, 1.0, 1.0, 1.0 };
  trackState.m_invNormFact = trackState.m_normFact;
  trackState.m_fourLayerSolution = false;

  for(auto& ah : eventState.m_allHits) ah.clear();

  //Find deflection values
  m_PrUTMagnetTool->dxNormFactorsUT( trackState.m_tyVelo,  trackState.m_normFact);
  std::transform(trackState.m_normFact.begin(),trackState.m_normFact.end(),trackState.m_invNormFact.begin(),
                 [](float normFact){return 1.0/normFact;});

  //Save some variables
  trackState.m_yAt0 = trackState.m_yVelo + trackState.m_tyVelo*(0. - trackState.m_zVelo);
  trackState.m_xMid = trackState.m_xVelo + trackState.m_txVelo*(m_zKink-trackState.m_zVelo);
  trackState.m_wb = m_sigmaVeloSlope*(m_zKink - trackState.m_zVelo);
  trackState.m_wb=1./(trackState.m_wb*trackState.m_wb);

  trackState.m_invKinkVeloDist = 1/(m_zKink-trackState.m_zVelo);

  //
  //Find VeloUT track candidates
  //
  auto c = getCandidate(velotrack, eventState, trackState);
  if (c){
    out.push_back(c.release());
    if( msgLevel(MSG::DEBUG) ){
      debug() << "CC and BC size: " << trackState.m_clusterCandidate.size() << " " << trackState.m_bestCandHits.size() << endmsg;
      for (auto  & hit : trackState.m_clusterCandidate) printHit(hit, "CC");
      for (auto  & hit : trackState.m_bestCandHits) printHit(hit, "BC");
    }
  }
  return StatusCode::SUCCESS;
}

//=========================================================================
// Get all the VeloUT track candidates
//=========================================================================
std::unique_ptr<LHCb::Track> PrVeloUTTool::getCandidate(const LHCb::Track& veloTrack,
                                                        PrVeloUTEventState &eventState,
                                                        PrVeloUTTrackState &trackState) const {

  // Find hits within a search window
  //--- it fills the m_hits
  if(!findHits(eventState, trackState)) return {};

  trackState.m_bestParams = { 0.0, m_maxPseudoChi2, 0.0, 0.0 };
  trackState.m_bestCandHits.clear();

  // -- Run clustering in forward direction
  formClusters(eventState, trackState);

  // -- Run clustering in backward direction
  if(!trackState.m_fourLayerSolution){
    std::reverse(eventState.m_allHits.begin(),eventState.m_allHits.end());
    formClusters(eventState, trackState);
    std::reverse(eventState.m_allHits.begin(),eventState.m_allHits.end());
  }

  //Write out the best solution
  return (!trackState.m_bestCandHits.empty()) ? prepareOutputTrack(veloTrack, eventState, trackState) : nullptr ;

}

//=========================================================================
//Find hits in a search window
//=========================================================================
bool PrVeloUTTool::findHits(PrVeloUTEventState &eventState,
                            PrVeloUTTrackState &trackState) const {

  // protect against unphysical angles, should not happen
  auto invTheta = std::min(500.,vdt::fast_isqrt(trackState.m_txVelo*trackState.m_txVelo+trackState.m_tyVelo*trackState.m_tyVelo));
  auto minP = ((m_minPT*invTheta)>m_minMomentum) ? (m_minPT*invTheta):m_minMomentum.value();
  auto xTol = std::abs(1. / ( m_distToMomentum * minP ));
  auto yTol = m_yTol + m_yTolSlope * xTol;

  unsigned int nHits = 0;

  //--------------------------------------------------------------------------
  // -- Loop on regions
  // -- If y > 0, only loop over upper half
  // -- If y < 0, only loop over lower half
  //--------------------------------------------------------------------------

  unsigned int startLoop = 0;
  unsigned int endLoop = 8;

  if( trackState.m_yAtMidUT > 0.0 ){
    startLoop = 4;
  }else{
    endLoop = 4;
  }


  for(unsigned int i = startLoop ; i < endLoop; ++i){
    if( (i == 6 || i == 2) && nHits == 0){
      return false;
    }

    //Protect against empty layers
    if(eventState.m_hitsLayers[i].empty()) continue;

    const float dxDy   = eventState.m_hitsLayers[i].front()->dxDy();
    float yLayer = 0.0;
    const float zLayer =  eventState.m_hitsLayers[i].front()->zAtYEq0();
    const float xLayer = trackState.m_xVelo + trackState.m_txVelo*(zLayer - trackState.m_zVelo);
    const float yAtZ   = trackState.m_yVelo + trackState.m_tyVelo*(zLayer - trackState.m_zVelo);

    yLayer =  yAtZ + std::copysign(yTol, yAtZ);

    // max distance between strips in a layer => 15mm
    // Hits are sorted at y=0
    const float lowerBoundX =
      xLayer - xTol*trackState.m_invNormFact[eventState.m_hitsLayers[i].front()->planeCode()] - dxDy*yLayer - fabs(trackState.m_txVelo)*m_intraLayerDist;

    if( msgLevel(MSG::DEBUG) ){
      debug() << "Loop : "<<i<<",  dxDy : "<<dxDy<<" yLayer : "<<yLayer<<" zLayer : "<<zLayer<<" yAtZ : "<<yAtZ<< endmsg;
      debug() << "===> LowerBoundX : "<<lowerBoundX<<"   (planeCode) "<<eventState.m_hitsLayers[i].front()->planeCode()<< endmsg;
    }

    const auto itEnd = eventState.m_hitsLayers[i].end();

    auto itH = std::lower_bound( eventState.m_hitsLayers[i].begin(), itEnd, lowerBoundX,
                                []( const UT::Hit* hit, double testval)->bool{
                                return  hit->xAtYEq0() < testval ;});

    if( msgLevel(MSG::DEBUG) ){
      debug() << "Size HL: " << eventState.m_hitsLayers[i].size() << ", Elements above lowerBoundX " << std::distance(itH,itEnd) << endmsg;
    }

    for ( ; itH != itEnd; ++itH ){

      if( msgLevel(MSG::DEBUG)){
        printHit( (*itH) , " xOnTrack loop ");
      }

      const float xOnTrack = trackState.m_xVelo + trackState.m_txVelo*((*itH)->zAtYEq0() - trackState.m_zVelo);

      //this used to be the hit update, now we calculate zz and xx here and test if we need the hit and only then create a MutUTHit
      //and store it for later use by the clustering etc.
      auto yy = (trackState.m_yAt0 + trackState.m_tyVelo * (*itH)->zAtYEq0() );

      auto zz = (*itH)->zAtYEq0();
      auto xx = (*itH)->xAt(yy);

      const float dx = xx - xOnTrack;

      // -- Scale to the reference reg
      const float normDx = dx * trackState.m_normFact[(*itH)->planeCode()];

      if( msgLevel(MSG::DEBUG) ){
        debug() << format("Updated x0:%8.2f, z0:%8.2f", xx, zz) << endmsg;
        debug() << " normDx  :"<< normDx <<"   xTol: "<<xTol<<endmsg;
      }

      if( normDx < -xTol ) continue;
      if( normDx > xTol ) break;

      const float fabsdx = std::abs(normDx);

      if(xTol > fabsdx){

        const float yOnTrack = trackState.m_yVelo + trackState.m_tyVelo*(zz - trackState.m_zVelo);

        // -- Now refine the tolerance in Y
        if( yOnTrack + (m_yTol + m_yTolSlope * fabsdx) < (*itH)->yMin() ||
            yOnTrack - (m_yTol + m_yTolSlope * fabsdx) > (*itH)->yMax() ) continue;

        if( msgLevel( MSG::DEBUG ) ){
          debug()<<"Push back in m_allHits["<<(*itH)->planeCode()<<"]"<<endmsg;
          printHit( (*itH));
        }

        eventState.m_allHits[(*itH)->planeCode()].emplace_back((*itH), xx, zz);
        ++nHits;

      }
    } // over hits
  }//over layers
  if( msgLevel(MSG::DEBUG)){
    debug()<<"nHits = "<<nHits<<endmsg;
  }
  return nHits > 2;
}
//=========================================================================
// Form clusters
//=========================================================================
void PrVeloUTTool::formClusters(PrVeloUTEventState &eventState,
                                PrVeloUTTrackState &trackState) const {

  if( msgLevel(MSG::DEBUG) ){
    debug()<<" Form Cluster Begin "<<endmsg;
    for( int i = 0 ; i<4; ++i){
      debug()<<"m_allHits["<<i<<"] size "<<eventState.m_allHits[i].size()<<endmsg;
    }
  }


  for(auto hit0 : eventState.m_allHits[0]){

    const float xhitLayer0 = hit0.x;
    const float zhitLayer0 = hit0.z;

    // Loop over Second Layer
    for(auto hit1 : eventState.m_allHits[1]){

      if(msgLevel(MSG::DEBUG)){
        printHit( hit1, " hit 1 ");
      }

      const float xhitLayer1 = hit1.x;
      const float zhitLayer1 = hit1.z;

      const float tx = (xhitLayer1 - xhitLayer0)/(zhitLayer1 - zhitLayer0);

      if( std::abs(tx-trackState.m_txVelo) > m_deltaTx1 ) continue;

      if(msgLevel(MSG::DEBUG)){
        debug()<<"PushBack hit0 and hit1 in m_clusterCandidate"<<endmsg;
      }

      trackState.m_clusterCandidate = { hit0, hit1 };

      for( auto hit2 : eventState.m_allHits[2]){

        if(msgLevel(MSG::DEBUG)){
          printHit( hit2, "  hit  2 ");
        }

        const float xhitLayer2 = hit2.x;
        const float zhitLayer2 = hit2.z;

        const float xextrapLayer2 = xhitLayer1 + tx*(zhitLayer2-zhitLayer1);
        if(std::abs(xhitLayer2 - xextrapLayer2) > m_hitTol1)continue;

        const float tx2 = (xhitLayer2 - xhitLayer0)/(zhitLayer2 - zhitLayer0);
        if(std::abs(tx2-trackState.m_txVelo)>m_deltaTx2) continue;

        if(msgLevel(MSG::DEBUG)){
          debug()<<"Push Back hit 2 in m_clusterCandidate"<<endmsg;
        }

        trackState.m_clusterCandidate.push_back(hit2);

        for( auto hit3 : eventState.m_allHits[3]){

          if(msgLevel(MSG::DEBUG)){
            printHit( hit3 , " hit 3 loop ");
          }
          const float xhitLayer3 = hit3.x;
          const float zhitLayer3 = hit3.z;

          const float xextrapLayer3 = xhitLayer2 + tx2*(zhitLayer3-zhitLayer2);

          if(std::abs(xhitLayer3 - xextrapLayer3) > m_hitTol2) continue;

          if(!trackState.m_fourLayerSolution){
            trackState.m_fourLayerSolution = true;
            trackState.m_bestParams = { 0.0, m_maxPseudoChi2, 0.0, 0.0 };
            trackState.m_bestCandHits.clear();
          }

          trackState.m_clusterCandidate.push_back( hit3 );
          if( msgLevel(MSG::DEBUG)){
            debug()<<"simpleFit<4> m_clusterCandidate"<<endmsg;
          }
          simpleFit<4>( trackState );
          trackState.m_clusterCandidate.pop_back();

        }//layer3

        if(!trackState.m_fourLayerSolution){

          if( msgLevel(MSG::DEBUG)){
            debug()<<"simpleFit<3> m_clusterCandidate"<<endmsg;
          }
          simpleFit<3>(trackState);

        }

        trackState.m_clusterCandidate.pop_back();
      }//layer2
      // Loop over Fourth Layer

      if(!trackState.m_fourLayerSolution){

        for( auto hit3 : eventState.m_allHits[3]){

          const float xhitLayer3 = hit3.x;
          const float zhitLayer3 = hit3.z;

          const float xextrapLayer3 = xhitLayer1 + tx*(zhitLayer3-zhitLayer1);
          if(std::abs(xhitLayer3 - xextrapLayer3) > m_hitTol1) continue;

          trackState.m_clusterCandidate.push_back(hit3);
          simpleFit<3>(trackState);
          trackState.m_clusterCandidate.pop_back();

        }
      }//layer3
    }
  }
}
//=========================================================================
// Create the Velo-TU tracks
//=========================================================================
std::unique_ptr<LHCb::Track> PrVeloUTTool::prepareOutputTrack(const LHCb::Track& veloTrack,
                                                              PrVeloUTEventState &eventState,
                                                              PrVeloUTTrackState &trackState) const {

  //== Handle states. copy Velo one, add TT.
  const float zOrigin = (std::fabs(trackState.m_tyVelo) > 0.001)
    ? trackState.m_zVelo - trackState.m_yVelo / trackState.m_tyVelo
    : trackState.m_zVelo - trackState.m_xVelo / trackState.m_txVelo;

  const float bdl= m_PrUTMagnetTool->bdlIntegral(trackState.m_tyVelo,zOrigin,trackState.m_zVelo);
  const float qpxz2p=-1*vdt::fast_isqrt(1.+trackState.m_tyVelo*trackState.m_tyVelo)/bdl*3.3356/Gaudi::Units::GeV;

  float qop = trackState.m_bestParams[0]*qpxz2p;
  if(std::abs(bdl) < 1.e-8 ) qop = 0.0;

  const float xUT  = trackState.m_bestParams[2];
  const float txUT = trackState.m_bestParams[3];

  std::unique_ptr<LHCb::Track> outTr{ new LHCb::Track()};

  // reset the track
  outTr->reset();
  outTr->copy(veloTrack);

  // Adding overlap hits
  for( auto hit : trackState.m_bestCandHits){

    outTr->addToLhcbIDs( hit.HitPtr->lhcbID() );

    const float xhit = hit.x;
    const float zhit = hit.z;

    for( auto ohit : eventState.m_allHits[hit.HitPtr->planeCode()]){
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

  //== Add a new state...
  LHCb::State temp;
  temp.setLocation( LHCb::State::AtTT );
  temp.setState( xUT,
                 trackState.m_yAtMidUT,
                 m_zMidUT,
                 txUT,
                 trackState.m_tyVelo,
                 qop );


  outTr->addToStates( std::move(temp) );
  outTr->setType( LHCb::Track::Upstream );
  outTr->setHistory( LHCb::Track::PatVeloTT );
  outTr->addToAncestors( veloTrack );
  outTr->setPatRecStatus( LHCb::Track::PatRecIDs );
  outTr->setChi2PerDoF( trackState.m_bestParams[1]);

  return outTr;
}
//=========================================================================
// createState
//=========================================================================
ranges::v3::any PrVeloUTTool::createState() const {
  PrVeloUTEventState eventState;

  const auto hh = m_HitHandler.get();

  for(int iStation = 0; iStation < 2; ++iStation){
    for(int iLayer = 0; iLayer < 2; ++iLayer){
      for( auto & hit : hh->hits( iStation, iLayer )){
        if( hit.yMax() > 0){
          eventState.m_hitsLayers[2*iStation + iLayer + 4].push_back( &hit);
        }
        if( hit.yMin() < 0){
          eventState.m_hitsLayers[ 2*iStation + iLayer].push_back( &hit );
        }
      }
    }
  }

  return eventState;
}
