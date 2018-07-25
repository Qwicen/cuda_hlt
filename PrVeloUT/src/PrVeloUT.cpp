#include "PrVeloUT.h"
//-----------------------------------------------------------------------------
// Implementation file for class : PrVeloUT
//
// 2007-05-08: Mariusz Witek
// 2017-03-01: Christoph Hasse (adapt to future framework)
// 2018-05-05: Plácido Fernández (make standalone)
//-----------------------------------------------------------------------------

namespace {
  // -- These things are all hardcopied from the PrTableForFunction
  // -- and PrUTMagnetTool
  // -- If the granularity or whatever changes, this will give wrong results

  int masterIndex(const int index1, const int index2, const int index3){
    return (index3*11 + index2)*31 + index1;
  }

  constexpr float minValsBdl[3] = { -0.3, -250.0, 0.0 };
  constexpr float maxValsBdl[3] = { 0.3, 250.0, 800.0 };
  constexpr float deltaBdl[3]   = { 0.02, 50.0, 80.0 };

  constexpr float dxDyHelper[4] = { 0.0, 1.0, -1.0, 0.0 };
}

//=============================================================================
// Initialization
//=============================================================================
// std::vector<std::string> PrVeloUT::GetFieldMaps() {
  
//   std::vector<std::string> filenames;
//   filenames.push_back("../PrUTMagnetTool/fieldmaps/field.v5r0.c1.down.cdf");
//   filenames.push_back("../PrUTMagnetTool/fieldmaps/field.v5r0.c2.down.cdf");
//   filenames.push_back("../PrUTMagnetTool/fieldmaps/field.v5r0.c3.down.cdf");
//   filenames.push_back("../PrUTMagnetTool/fieldmaps/field.v5r0.c4.down.cdf");

//   return filenames;
// }

int PrVeloUT::initialize() {

  //load the deflection and Bdl values from a text file
  float dxLayTable[PrUTMagnetTool::N_dxLay_vals];
  std::ifstream deflectionfile;
  deflectionfile.open("../PrUTMagnetTool/deflection.txt");
  if (deflectionfile.is_open()) {
    int i = 0;
    float deflection;
    while (!deflectionfile.eof()) {
      deflectionfile >> deflection;
      assert( i < PrUTMagnetTool::N_dxLay_vals );
      dxLayTable[i++] = deflection;
    }
  }
  
  float bdlTable[PrUTMagnetTool::N_bdl_vals];
  std::ifstream bdlfile;
  bdlfile.open("../PrUTMagnetTool/bdl.txt");
  if (bdlfile.is_open()) {
    int i = 0;
    float bdl;
    while (!bdlfile.eof()) {
      bdlfile >> bdl;
      assert( i < PrUTMagnetTool::N_bdl_vals );
      bdlTable[i++] = bdl;
    }
  }
  
  m_PrUTMagnetTool = PrUTMagnetTool( dxLayTable, bdlTable );
  
  // m_zMidUT is a position of normalization plane which should to be close to z middle of UT ( +- 5 cm ).
  // Cashed once in PrVeloUTTool at initialization. No need to update with small UT movement.
  m_zMidUT    = 2484.6;
  //  zMidField and distToMomentum isproperly recalculated in PrUTMagnetTool when B field changes
  m_distToMomentum = 4.0212e-05;

  m_sigmaVeloSlope = 0.10*Gaudi::Units::mrad;
  m_invSigmaVeloSlope = 1.0/m_sigmaVeloSlope;
  m_zKink = 1780.0;

  return 1;
}

//=============================================================================
// Main execution
//=============================================================================
std::vector<VeloUTTracking::TrackVeloUT> PrVeloUT::operator() (
  const std::vector<VeloUTTracking::TrackVelo>& inputTracks,
  VeloUTTracking::HitsSoA *hits_layers,
  const uint32_t n_hits_layers[VeloUTTracking::n_layers]
  ) const
{
  
  std::vector<VeloUTTracking::TrackVeloUT> outputTracks;
  outputTracks.reserve(inputTracks.size());

  std::array<std::array<int,85>,4> posLayers;
  fillIterators(hits_layers, n_hits_layers, posLayers);

  const float* fudgeFactors = m_PrUTMagnetTool.returnDxLayTable();
  const float* bdlTable     = m_PrUTMagnetTool.returnBdlTable();

  // array to store indices of selected hits in layers
  // -> can then access the hit information in the HitsSoA
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers];
  
  for(const VeloUTTracking::TrackVelo& veloTr : inputTracks) {
    
    VeloState trState;
    if( !getState(veloTr, trState)) continue;
    for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      n_hitCandidatesInLayers[i_layer] = 0;
    }
    if( !getHits(hitCandidatesInLayers, n_hitCandidatesInLayers, posLayers, hits_layers, n_hits_layers, fudgeFactors, trState ) ) continue;

    TrackHelper helper(trState, m_zKink, m_sigmaVeloSlope, m_maxPseudoChi2);

    // go through UT layers in forward direction
    if( !formClusters(hitCandidatesInLayers, n_hitCandidatesInLayers, hits_layers, helper, true) ){

      // go through UT layers in backward direction
      formClusters(hitCandidatesInLayers, n_hitCandidatesInLayers, hits_layers, helper, false);
    }

    if ( helper.n_hits > 0 ) {
      prepareOutputTrack(veloTr, helper, hitCandidatesInLayers, n_hitCandidatesInLayers, hits_layers, outputTracks, bdlTable);
    }
  }

  return outputTracks;
}

//=============================================================================
// Get the state, do some cuts
//=============================================================================
bool PrVeloUT::getState(
  const VeloUTTracking::TrackVelo& iTr, 
  VeloState& trState ) const 
{
  const VeloState state = iTr.state;
  
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
    return false;
  }

  return true;

}

//=============================================================================
// Find the hits
//=============================================================================
bool PrVeloUT::getHits(
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],		       
  const std::array<std::array<int,85>,4>& posLayers,
  VeloUTTracking::HitsSoA *hits_layers,
  const uint32_t n_hits_layers[VeloUTTracking::n_layers],
  const float* fudgeFactors, 
  VeloState& trState ) const 
{
  // -- This is hardcoded, so faster
  // -- If you ever change the Table in the magnet tool, this will be wrong
  const float absSlopeY = std::abs( trState.ty );
  const int index = (int)(absSlopeY*100 + 0.5);
  assert( 3 + 4*index < PrUTMagnetTool::N_dxLay_vals );
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
      int layer_offset = hits_layers->layer_offset[layer];
      
      if( n_hits_layers[layer] == 0 ) continue;

      const float dxDy   = hits_layers->dxDy(layer_offset + 0);
      const float zLayer = hits_layers->zAtYEq0(layer_offset + 0); 

      const float yAtZ   = trState.y + trState.ty*(zLayer - trState.z);
      const float xLayer = trState.x + trState.tx*(zLayer - trState.z);
      const float yLayer = yAtZ + yTol*dxDyHelper[2*iStation+iLayer];

      const float normFactNum = normFact[2*iStation + iLayer];
      const float invNormFact = 1.0/normFactNum;

      const float lowerBoundX =
        (xLayer - dxDy*yLayer) - xTol*invNormFact - std::abs(trState.tx)*m_intraLayerDist;
      const float upperBoundX =
        (xLayer - dxDy*yLayer) + xTol*invNormFact + std::abs(trState.tx)*m_intraLayerDist;

      const int indexLowProto = lowerBoundX > 0 ? std::sqrt( std::abs(lowerBoundX)*2.0 ) + 42 : 42 - std::sqrt( std::abs(lowerBoundX)*2.0 );
      const int indexHiProto  = upperBoundX > 0 ? std::sqrt( std::abs(upperBoundX)*2.0 ) + 43 : 43 - std::sqrt( std::abs(upperBoundX)*2.0 );

      const int indexLow  = std::max( indexLowProto, 0 );
      const int indexHi   = std::min( indexHiProto, 84);

      size_t posBeg = posLayers[layer][ indexLow ];
      size_t posEnd = posLayers[layer][ indexHi  ];

      while ( (hits_layers->xAtYEq0(layer_offset + posBeg) < lowerBoundX) && (posBeg != n_hits_layers[layer] ) )
	++posBeg;
      if (posBeg == n_hits_layers[layer]) continue;

      findHits(posBeg, posEnd, hits_layers, n_hits_layers, layer_offset, trState, xTol*invNormFact, invNormFact, hitCandidatesInLayers[layer], n_hitCandidatesInLayers[layer]);

      nLayers += int( !( n_hitCandidatesInLayers[layer] == 0 ) );
    }
  }

  return nLayers > 2;
}

//=========================================================================
// Form clusters
//=========================================================================
bool PrVeloUT::formClusters(
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  VeloUTTracking::HitsSoA *hits_layers,
  TrackHelper& helper,
  const bool forward ) const 
{

  // handle forward / backward cluster search
  int layers[VeloUTTracking::n_layers];
  for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
      if ( forward )
        layers[i_layer] = i_layer;
      else
        layers[i_layer] = VeloUTTracking::n_layers - 1 - i_layer;
  }

  // Go through the layers
  bool fourLayerSolution = false;
  for ( int i_hit0 = 0; i_hit0 < n_hitCandidatesInLayers[ layers[0] ]; ++i_hit0 ) {

    const int hit_index0    = hitCandidatesInLayers[ layers[0] ][i_hit0];
    const int layer_offset0 = hits_layers->layer_offset[ layers[0] ];
    const float xhitLayer0  = hits_layers->x[ layer_offset0 + hit_index0 ];
    const float zhitLayer0  = hits_layers->z[ layer_offset0 + hit_index0 ];
    
    for ( int i_hit2 = 0; i_hit2 < n_hitCandidatesInLayers[ layers[2] ]; ++i_hit2 ) {
      
      const int hit_index2    = hitCandidatesInLayers[ layers[2] ][i_hit2];
      const int layer_offset2 = hits_layers->layer_offset[ layers[2] ];
      const float xhitLayer2  = hits_layers->x[ layer_offset2 + hit_index2 ];
      const float zhitLayer2  = hits_layers->z[ layer_offset2 + hit_index2 ];
       
      const float tx = (xhitLayer2 - xhitLayer0)/(zhitLayer2 - zhitLayer0);
      if( std::abs(tx-helper.state.tx) > m_deltaTx2 ) continue;
      
      int IndexBestHit1 = -10;
      float hitTol = m_hitTol2;
      for ( int i_hit1 = 0; i_hit1 < n_hitCandidatesInLayers[ layers[1] ]; ++i_hit1 ) {

        const int hit_index1    = hitCandidatesInLayers[ layers[1] ][i_hit1];
        const int layer_offset1 = hits_layers->layer_offset[ layers[1] ];
        const float xhitLayer1  = hits_layers->x[ layer_offset1 + hit_index1 ];
        const float zhitLayer1  = hits_layers->z[ layer_offset1 + hit_index1 ];
       

        const float xextrapLayer1 = xhitLayer0 + tx*(zhitLayer1-zhitLayer0);
        if(std::abs(xhitLayer1 - xextrapLayer1) < hitTol){
          hitTol = std::abs(xhitLayer1 - xextrapLayer1);
          IndexBestHit1 = hit_index1;
        }
      } // loop over layer 1
      VeloUTTracking::Hit bestHit1;
      if ( IndexBestHit1 > 0 ) { // found hit candidate
        bestHit1 = VeloUTTracking::createHit(hits_layers, layers[1], IndexBestHit1);
      }
      
      if( fourLayerSolution && IndexBestHit1 < 0 ) continue;

      int IndexBestHit3 = -10;
      hitTol = m_hitTol2;
      for ( int i_hit3 = 0; i_hit3 < n_hitCandidatesInLayers[ layers[3] ]; ++i_hit3 ) {

        const int hit_index3    = hitCandidatesInLayers[ layers[3] ][i_hit3];
        const int layer_offset3 = hits_layers->layer_offset[ layers[3] ];
        const float xhitLayer3  = hits_layers->x[ layer_offset3 + hit_index3 ];
        const float zhitLayer3  = hits_layers->z[ layer_offset3 + hit_index3 ];

        const float xextrapLayer3 = xhitLayer2 + tx*(zhitLayer3-zhitLayer2);
        if(std::abs(xhitLayer3 - xextrapLayer3) < hitTol){
          hitTol = std::abs(xhitLayer3 - xextrapLayer3);
          IndexBestHit3 = hit_index3;
        }
      } // loop over layer 3
      VeloUTTracking::Hit bestHit3;
      if ( IndexBestHit3 > 0 ) { // found hit candidate
        bestHit3 = VeloUTTracking::createHit(hits_layers, layers[3], IndexBestHit3);
      }

      // -- All hits found
      if ( IndexBestHit1 > 0 && IndexBestHit3 > 0 ) {
        VeloUTTracking::Hit hit0 = VeloUTTracking::createHit(hits_layers, layers[0], hit_index0);
        VeloUTTracking::Hit hit2 = VeloUTTracking::createHit(hits_layers, layers[2], hit_index2);
        std::array<const VeloUTTracking::Hit*,4> hits4fit = {&hit0, &bestHit1, &hit2, &bestHit3};
        simpleFit(hits4fit, helper);
        
        if(!fourLayerSolution && helper.n_hits > 0){
          fourLayerSolution = true;
        }
        continue;
      }

      // -- Nothing found in layer 3
      if( !fourLayerSolution && IndexBestHit1 > 0 ){
        VeloUTTracking::Hit hit0 = VeloUTTracking::createHit(hits_layers, layers[0], hit_index0);
        VeloUTTracking::Hit hit2 = VeloUTTracking::createHit(hits_layers, layers[2], hit_index2);
        std::array<const VeloUTTracking::Hit*,3> hits4fit = {&hit0, &bestHit1, &hit2};
        simpleFit(hits4fit,  helper);
        continue;
      }
      // -- Nothing found in layer 1
      if( !fourLayerSolution && IndexBestHit3 > 0 ){
        VeloUTTracking::Hit hit0 = VeloUTTracking::createHit(hits_layers, layers[0], hit_index0);
        VeloUTTracking::Hit hit2 = VeloUTTracking::createHit(hits_layers, layers[2], hit_index2);
        std::array<const VeloUTTracking::Hit*,3> hits4fit = {&hit0, &bestHit3, &hit2};
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
void PrVeloUT::prepareOutputTrack(
  const VeloUTTracking::TrackVelo& veloTrack,
  const TrackHelper& helper,
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  VeloUTTracking::HitsSoA *hits_layers,
  std::vector<VeloUTTracking::TrackVeloUT>& outputTracks,
  const float* bdlTable) const {

  //== Handle states. copy Velo one, add TT.
  const float zOrigin = (std::fabs(helper.state.ty) > 0.001)
    ? helper.state.z - helper.state.y / helper.state.ty
    : helper.state.z - helper.state.x / helper.state.tx;

  // -- These are calculations, copied and simplified from PrTableForFunction
  const std::array<float,3> var = { helper.state.ty, zOrigin, helper.state.z };

  const int index1 = std::max(0, std::min( 30, int((var[0] + 0.3)/0.6*30) ));
  const int index2 = std::max(0, std::min( 10, int((var[1] + 250)/500*10) ));
  const int index3 = std::max(0, std::min( 10, int( var[2]/800*10)        ));

  assert( masterIndex(index1, index2, index3) < PrUTMagnetTool::N_bdl_vals );
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

  VeloUTTracking::TrackVeloUT outputtrack; 
  outputtrack.track = veloTrack.track;
  outputTracks.emplace_back( outputtrack );
  
  // Adding overlap hits
  for ( int i_hit = 0; i_hit < helper.n_hits; ++i_hit ) {
    const VeloUTTracking::Hit hit = helper.bestHits[i_hit];
    
    outputTracks.back().track.addLHCbID( hit.LHCbID() );

    const float xhit = hit.x;
    const float zhit = hit.z;

    const int planeCode = hit.planeCode();
    for ( int i_ohit = 0; i_ohit < n_hitCandidatesInLayers[planeCode]; ++i_ohit ) {
      const int ohit_index = hitCandidatesInLayers[planeCode][i_ohit];
      const int layer_offset = hits_layers->layer_offset[planeCode];
      
      const float zohit = hits_layers->z[layer_offset + ohit_index];
      if(zohit==zhit) continue;

      const float xohit = hits_layers->x[layer_offset + ohit_index];
      const float xextrap = xhit + txUT*(zhit-zohit);
      if( xohit-xextrap < -m_overlapTol) continue;
      if( xohit-xextrap > m_overlapTol) break;
    
      outputTracks.back().track.addLHCbID( hits_layers->LHCbID(layer_offset + ohit_index) );

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

  
}


