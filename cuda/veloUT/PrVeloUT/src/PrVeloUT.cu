#include "PrVeloUT.cuh"

//-----------------------------------------------------------------------------
// Implementation file for PrVeloUT
//
// 2007-05-08: Mariusz Witek
// 2017-03-01: Christoph Hasse (adapt to future framework)
// 2018-05-05: Plácido Fernández (make standalone)
// 2018-07:    Dorothea vom Bruch (convert to C and then CUDA code)
//-----------------------------------------------------------------------------

// -- These things are all hardcopied from the PrTableForFunction
// -- and PrUTMagnetTool
// -- If the granularity or whatever changes, this will give wrong results

  __host__ __device__ int masterIndex(const int index1, const int index2, const int index3){
    return (index3*11 + index2)*31 + index1;
  }

 


//=============================================================================
// Reject tracks outside of acceptance or pointing to the beam pipe
//=============================================================================
__host__ __device__ bool veloTrackInUTAcceptance( const VeloState& state )
{

  const float xMidUT =  state.x + state.tx*( PrVeloUTConst::zMidUT - state.z);
  const float yMidUT =  state.y + state.ty*( PrVeloUTConst::zMidUT - state.z);

  if( xMidUT*xMidUT+yMidUT*yMidUT  < PrVeloUTConst::centralHoleSize*PrVeloUTConst::centralHoleSize ) return false;
  if( (std::abs(state.tx) > PrVeloUTConst::maxXSlope) || (std::abs(state.ty) > PrVeloUTConst::maxYSlope) ) return false;

  if(PrVeloUTConst::passTracks && std::abs(xMidUT) < PrVeloUTConst::passHoleSize && std::abs(yMidUT) < PrVeloUTConst::passHoleSize) {
    return false;
  }

  return true;
}

//=============================================================================
// Find the hits
//=============================================================================
__host__ __device__ bool getHits(
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int posLayers[4][85],
  VeloUTTracking::HitsSoA *hits_layers,
  const float* fudgeFactors, 
  const VeloState& trState )
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
  // to do: change back!
  const float invTheta = std::min(500.,1.0/std::sqrt(trState.tx*trState.tx+trState.ty*trState.ty));
  //const float minMom   = std::max(PrVeloUTConst::minPT*invTheta, PrVeloUTConst::minMomentum);
  const float minMom   = std::max(PrVeloUTConst::minPT*invTheta, float(1.5e3));
  const float xTol     = std::abs(1. / ( PrVeloUTConst::distToMomentum * minMom ));
  const float yTol     = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * xTol;

  int nLayers = 0;

  for(int iStation = 0; iStation < 2; ++iStation) {

    if( iStation == 1 && nLayers == 0 ) return false;

    for(int iLayer = 0; iLayer < 2; ++iLayer) {
      if( iStation == 1 && iLayer == 1 && nLayers < 2 ) return false;

      int layer = 2*iStation+iLayer;
      int layer_offset = hits_layers->layer_offset[layer];
      
      if( hits_layers->n_hits_layers[layer] == 0 ) continue;

      const float dxDy   = hits_layers->dxDy(layer_offset + 0);
      const float zLayer = hits_layers->zAtYEq0(layer_offset + 0); 

      const float yAtZ   = trState.y + trState.ty*(zLayer - trState.z);
      const float xLayer = trState.x + trState.tx*(zLayer - trState.z);
#ifdef __CUDA_ARCH__
      const float yLayer = yAtZ + yTol*PrVeloUTConst::dev_dxDyHelper[2*iStation+iLayer];
#else
      const float yLayer = yAtZ + yTol*PrVeloUTConst::dxDyHelper[2*iStation+iLayer];
#endif

      const float normFactNum = normFact[2*iStation + iLayer];
      const float invNormFact = 1.0/normFactNum;

      const float lowerBoundX =
        (xLayer - dxDy*yLayer) - xTol*invNormFact - std::abs(trState.tx)*PrVeloUTConst::intraLayerDist;
      const float upperBoundX =
        (xLayer - dxDy*yLayer) + xTol*invNormFact + std::abs(trState.tx)*PrVeloUTConst::intraLayerDist;

      const int indexLowProto = lowerBoundX > 0 ? std::sqrt( std::abs(lowerBoundX)*2.0 ) + 42 : 42 - std::sqrt( std::abs(lowerBoundX)*2.0 );
      const int indexHiProto  = upperBoundX > 0 ? std::sqrt( std::abs(upperBoundX)*2.0 ) + 43 : 43 - std::sqrt( std::abs(upperBoundX)*2.0 );

      const int indexLow  = std::max( indexLowProto, 0 );
      const int indexHi   = std::min( indexHiProto, 84);

      size_t posBeg = posLayers[layer][ indexLow ];
      size_t posEnd = posLayers[layer][ indexHi  ];

      while ( (hits_layers->xAtYEq0(layer_offset + posBeg) < lowerBoundX) && (posBeg != hits_layers->n_hits_layers[layer] ) )
	++posBeg;
      if (posBeg == hits_layers->n_hits_layers[layer]) continue;

      findHits(posBeg, posEnd,
        hits_layers, layer_offset, layer,
        trState, xTol*invNormFact, invNormFact,
        hitCandidatesInLayers[layer], n_hitCandidatesInLayers[layer],
        x_pos_layers);

      nLayers += int( !( n_hitCandidatesInLayers[layer] == 0 ) );
    }
  }

  return nLayers > 2;
}

//=========================================================================
// Form clusters
//=========================================================================
__host__ __device__ bool formClusters(
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int bestHitCandidateIndices[VeloUTTracking::n_layers],
  VeloUTTracking::HitsSoA *hits_layers,
  TrackHelper& helper,
  const bool forward)
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
  int hitCandidateIndices[VeloUTTracking::n_layers];
  for ( int i_hit0 = 0; i_hit0 < n_hitCandidatesInLayers[ layers[0] ]; ++i_hit0 ) {

    const int hit_index0    = hitCandidatesInLayers[ layers[0] ][i_hit0];
    const float xhitLayer0  = x_pos_layers[layers[0]][i_hit0];
    const int layer_offset0 = hits_layers->layer_offset[ layers[0] ];
    const float zhitLayer0  = hits_layers->zAtYEq0( layer_offset0 + hit_index0 );
    hitCandidateIndices[0] = i_hit0;
    
    for ( int i_hit2 = 0; i_hit2 < n_hitCandidatesInLayers[ layers[2] ]; ++i_hit2 ) {
     
      const int hit_index2    = hitCandidatesInLayers[ layers[2] ][i_hit2];
      const float xhitLayer2  = x_pos_layers[layers[2]][i_hit2];
      const int layer_offset2 = hits_layers->layer_offset[ layers[2] ];
      const float zhitLayer2  = hits_layers->zAtYEq0( layer_offset2 + hit_index2 );
      hitCandidateIndices[2] = i_hit2;
      
      const float tx = (xhitLayer2 - xhitLayer0)/(zhitLayer2 - zhitLayer0);
      if( std::abs(tx-helper.state.tx) > PrVeloUTConst::deltaTx2 ) continue;
            
      int IndexBestHit1 = -10;
      float hitTol = PrVeloUTConst::hitTol2;
      for ( int i_hit1 = 0; i_hit1 < n_hitCandidatesInLayers[ layers[1] ]; ++i_hit1 ) {

        const int hit_index1    = hitCandidatesInLayers[ layers[1] ][i_hit1];
        const float xhitLayer1  = x_pos_layers[layers[1]][i_hit1];
        const int layer_offset1 = hits_layers->layer_offset[ layers[1] ];
        const float zhitLayer1  = hits_layers->zAtYEq0( layer_offset1 + hit_index1 );
       
        const float xextrapLayer1 = xhitLayer0 + tx*(zhitLayer1-zhitLayer0);
        if(std::abs(xhitLayer1 - xextrapLayer1) < hitTol){
          hitTol = std::abs(xhitLayer1 - xextrapLayer1);
          IndexBestHit1 = hit_index1;
          hitCandidateIndices[1] = i_hit1;
        }
      } // loop over layer 1
      VeloUTTracking::Hit bestHit1;
      if ( IndexBestHit1 > 0 ) { // found hit candidate
        bestHit1 = VeloUTTracking::createHit(hits_layers, layers[1], IndexBestHit1);
      }
      
      if( fourLayerSolution && IndexBestHit1 < 0 ) continue;

      int IndexBestHit3 = -10;
      hitTol = PrVeloUTConst::hitTol2;
      for ( int i_hit3 = 0; i_hit3 < n_hitCandidatesInLayers[ layers[3] ]; ++i_hit3 ) {

        const int hit_index3    = hitCandidatesInLayers[ layers[3] ][i_hit3];
        const float xhitLayer3  = x_pos_layers[layers[3]][i_hit3];
        const int layer_offset3 = hits_layers->layer_offset[ layers[3] ];
        const float zhitLayer3  = hits_layers->zAtYEq0( layer_offset3 + hit_index3 );
        
        const float xextrapLayer3 = xhitLayer2 + tx*(zhitLayer3-zhitLayer2);
        if(std::abs(xhitLayer3 - xextrapLayer3) < hitTol){
          hitTol = std::abs(xhitLayer3 - xextrapLayer3);
          IndexBestHit3 = hit_index3;
          hitCandidateIndices[3] = i_hit3;
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
        const VeloUTTracking::Hit* hits4fit[4] = {&hit0, &bestHit1, &hit2, &bestHit3};
        simpleFit<4>(hits4fit, x_pos_layers, hitCandidateIndices, bestHitCandidateIndices, hitCandidatesInLayers, hits_layers, helper);
        
        if(!fourLayerSolution && helper.n_hits > 0){
          fourLayerSolution = true;
        }
        continue;
      }

      // -- Nothing found in layer 3
      if( !fourLayerSolution && IndexBestHit1 > 0 ){
        VeloUTTracking::Hit hit0 = VeloUTTracking::createHit(hits_layers, layers[0], hit_index0);
        VeloUTTracking::Hit hit2 = VeloUTTracking::createHit(hits_layers, layers[2], hit_index2);
        const VeloUTTracking::Hit* hits4fit[3] = {&hit0, &bestHit1, &hit2};
        simpleFit<3>(hits4fit, x_pos_layers, hitCandidateIndices, bestHitCandidateIndices, hitCandidatesInLayers, hits_layers, helper);
        continue;
      }
      // -- Nothing found in layer 1
      if( !fourLayerSolution && IndexBestHit3 > 0 ){
        hitCandidateIndices[1] = hitCandidateIndices[3];  // hit3 saved in second position of hits4fit
        VeloUTTracking::Hit hit0 = VeloUTTracking::createHit(hits_layers, layers[0], hit_index0);
        VeloUTTracking::Hit hit2 = VeloUTTracking::createHit(hits_layers, layers[2], hit_index2);
        const VeloUTTracking::Hit* hits4fit[3] = {&hit0, &bestHit3, &hit2};
        simpleFit<3>(hits4fit, x_pos_layers, hitCandidateIndices, bestHitCandidateIndices, hitCandidatesInLayers, hits_layers, helper);
        continue;
      }
      
    }
  }

  return fourLayerSolution;
}
//=========================================================================
// Create the Velo-TU tracks
//=========================================================================
__host__ __device__ void prepareOutputTrack(
  const uint* velo_track_hit_number,   
  const VeloTracking::Hit<mc_check_enabled>* velo_track_hits,
  const int accumulated_tracks_event,
  const int i_Velo_track,
  const TrackHelper& helper,
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  VeloUTTracking::HitsSoA *hits_layers,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks],
  int* n_veloUT_tracks,
  const float* bdlTable) {

  //== Handle states. copy Velo one, add UT.
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

  const float bdls[3] = { bdlTable[masterIndex(index1+1, index2,index3)],
                          bdlTable[masterIndex(index1,index2+1,index3)],
                          bdlTable[masterIndex(index1,index2,index3+1)] };

  const float boundaries[3] = { -0.3f + float(index1)*PrVeloUTConst::deltaBdl[0],
                                -250.0f + float(index2)*PrVeloUTConst::deltaBdl[1],
                                0.0f + float(index3)*PrVeloUTConst::deltaBdl[2] };

  // -- This is an interpolation, to get a bit more precision
  float addBdlVal = 0.0;
  for(int i=0; i<3; ++i) {
#ifdef __CUDA_ARCH__
    if( var[i] < PrVeloUTConst::dev_minValsBdl[i] || var[i] > PrVeloUTConst::dev_maxValsBdl[i] ) continue;
    const float dTab_dVar =  (bdls[i] - bdl) / PrVeloUTConst::dev_deltaBdl[i];
#else
    if( var[i] < PrVeloUTConst::minValsBdl[i] || var[i] > PrVeloUTConst::maxValsBdl[i] ) continue;
    const float dTab_dVar =  (bdls[i] - bdl) / PrVeloUTConst::deltaBdl[i];
#endif
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

  if( p < PrVeloUTConst::minMomentum || pt < PrVeloUTConst::minPT ) return;

#ifdef __CUDA_ARCH__
  uint n_tracks = atomicAdd(n_veloUT_tracks, 1);
  //assert( n_tracks < VeloUTTracking::max_num_tracks );
 // #else
//   assert( *n_veloUT_tracks < VeloUTTracking::max_num_tracks );
//   VeloUT_tracks[*n_veloUT_tracks] = track;
//   (*n_veloUT_tracks)++;
#endif

  
  const float txUT = helper.bestParams[3];

#ifdef MC_CHECK
  VeloUTTracking::TrackUT track;
  track.hitsNum = 0;
  const uint starting_hit = velo_track_hit_number[accumulated_tracks_event + i_Velo_track];
  const uint number_of_hits = velo_track_hit_number[accumulated_tracks_event + i_Velo_track + 1]  - starting_hit;
  for ( int i_hit = 0; i_hit < number_of_hits; ++i_hit ) {
    track.addLHCbID( velo_track_hits[starting_hit + i_hit].LHCbID );
    assert( track.hitsNum < VeloUTTracking::max_track_size);
  }
  track.set_qop( qop );
  
  // Adding overlap hits
  for ( int i_hit = 0; i_hit < helper.n_hits; ++i_hit ) {
    const VeloUTTracking::Hit hit = helper.bestHits[i_hit];
    
    track.addLHCbID( hit.LHCbID() );
    assert( track.hitsNum < VeloUTTracking::max_track_size);
    
    const int planeCode = hit.planeCode();
    const float xhit = x_pos_layers[ planeCode ][ hitCandidateIndices[i_hit] ];
    const int layer_offset = hits_layers->layer_offset[ planeCode ];
    const int hit_index = hitCandidatesInLayers[planeCode][ hitCandidateIndices[i_hit] ];
    const float zhit = hits_layers->zAtYEq0( layer_offset + hit_index );
    
    for ( int i_ohit = 0; i_ohit < n_hitCandidatesInLayers[planeCode]; ++i_ohit ) {
      const int ohit_index = hitCandidatesInLayers[planeCode][i_ohit];
      const float zohit  = hits_layers->zAtYEq0( layer_offset + ohit_index );
      
      if(zohit==zhit) continue;
      
      const float xohit = x_pos_layers[ planeCode ][ i_ohit];
      const float xextrap = xhit + txUT*(zhit-zohit);
      if( xohit-xextrap < -PrVeloUTConst::overlapTol) continue;
      if( xohit-xextrap > PrVeloUTConst::overlapTol) break;
      
      track.addLHCbID( hits_layers->LHCbID(layer_offset + ohit_index) );
      assert( track.hitsNum < VeloUTTracking::max_track_size);
      
      // -- only one overlap hit
      break;
    }
  }
#ifdef __CUDA_ARCH__
  assert( n_tracks < VeloUTTracking::max_num_tracks );
  VeloUT_tracks[n_tracks] = track;
#endif
#endif

  /*
  outTr.x = helper.state.x;
  outTr.y = helper.state.y;
  outTr.z = helper.state.z;
  outTr.tx = helper.state.tx;
  outTr.ty = helper.state.ty;
  */
}

__host__ __device__ void fillArray(
  int * array,
  const int size,
  const size_t value ) {
  for ( int i = 0; i < size; ++i ) {
    array[i] = value;
  }
}

__host__ __device__ void fillArrayAt(
  int * array,
  const int offset,
  const int n_vals,
  const size_t value ) {  
    fillArray( array + offset, n_vals, value ); 
}

// ==============================================================================
// -- Method to cache some starting points for the search
// -- This is actually faster than binary searching the full array
// -- Granularity hardcoded for the moment.
// -- Idea is: UTb has dimensions in x (at y = 0) of about -860mm -> 860mm
// -- The indices go from 0 -> 84, and shift by -42, leading to -42 -> 42
// -- Taking the higher density of hits in the center into account, the positions of the iterators are
// -- calculated as index*index/2, where index = [ -42, 42 ], leading to
// -- -882mm -> 882mm
// -- The last element is an "end" iterator, to make sure we never go out of bound
// ==============================================================================
__host__ __device__ void fillIterators(
  VeloUTTracking::HitsSoA *hits_layers,
  int posLayers[4][85] )
{
    
  for(int iStation = 0; iStation < 2; ++iStation){
    for(int iLayer = 0; iLayer < 2; ++iLayer){
      int layer = 2*iStation + iLayer;
      int layer_offset = hits_layers->layer_offset[layer];
      
      size_t pos = 0;
      // to do: check whether there is an efficient thrust implementation for this
      fillArray( posLayers[layer], 85, pos );
      
      int bound = -42.0;
      // to do : make copysignf
      float val = std::copysign(float(bound*bound)/2.0, bound);
      
      // TODO add bounds checking
      for ( ; pos != hits_layers->n_hits_layers[layer]; ++pos) {
        while( hits_layers->xAtYEq0( layer_offset + pos ) > val){
          posLayers[layer][bound+42] = pos;
          ++bound;
          val = std::copysign(float(bound*bound)/2.0, bound);
        }
      }
      
      fillArrayAt(
        posLayers[layer],
        42 + bound,
        85 - 42 - bound,
        hits_layers->n_hits_layers[layer] );
      
    }
  }
}


// ==============================================================================
// -- Finds the hits in a given layer within a certain range
// ==============================================================================
__host__ __device__ void findHits( 
  const size_t posBeg,
  const size_t posEnd,
  VeloUTTracking::HitsSoA *hits_layers,
  const int layer_offset,
  // to do: pass array for this layer -> get rid of i_layer index
  const int i_layer,
  const VeloState& myState, 
  const float xTolNormFact,
  const float invNormFact,
  int hitCandidatesInLayer[VeloUTTracking::max_hit_candidates_per_layer],
  int &n_hitCandidatesInLayer,
  float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer])
{
  const auto zInit = hits_layers->zAtYEq0( layer_offset + posBeg );
  const auto yApprox = myState.y + myState.ty * (zInit - myState.z);
  
  size_t pos = posBeg;
  while ( 
   pos <= posEnd && 
   hits_layers->isNotYCompatible( layer_offset + pos, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(xTolNormFact) )
   ) { ++pos; }

  const auto xOnTrackProto = myState.x + myState.tx*(zInit - myState.z);
  const auto yyProto =       myState.y - myState.ty*myState.z;
  
  for (int i=pos; i<posEnd; ++i) {
    
    const auto xx = hits_layers->xAt( layer_offset + i, yApprox ); 
    const auto dx = xx - xOnTrackProto;
    
    if( dx < -xTolNormFact ) continue;
    if( dx >  xTolNormFact ) break; 
    
    // -- Now refine the tolerance in Y
    if ( hits_layers->isNotYCompatible( layer_offset + i, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx*invNormFact)) ) continue;
    
    const auto zz = hits_layers->zAtYEq0( layer_offset + i ); 
    const auto yy = yyProto +  myState.ty*zz;
    const auto xx2 = hits_layers->xAt( layer_offset + i, yy );
        
    hitCandidatesInLayer[n_hitCandidatesInLayer] = i;
    x_pos_layers[i_layer][n_hitCandidatesInLayer] = xx2;
    
    n_hitCandidatesInLayer++;

    if ( n_hitCandidatesInLayer >= VeloUTTracking::max_hit_candidates_per_layer )
      printf("%u > %u !! \n", n_hitCandidatesInLayer, VeloUTTracking::max_hit_candidates_per_layer);
    assert( n_hitCandidatesInLayer < VeloUTTracking::max_hit_candidates_per_layer );
  }
  for ( int i_hit = 0; i_hit < n_hitCandidatesInLayer; ++i_hit ) {
    assert( hitCandidatesInLayer[i_hit] < VeloUTTracking::max_numhits_per_event );
  }
}
 
// =================================================
// -- 2 helper functions for fit
// -- Pseudo chi2 fit, templated for 3 or 4 hits
// =================================================
__host__ __device__ void addHits(
  float* mat,
  float* rhs,
  const VeloUTTracking::Hit** hits,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  VeloUTTracking::HitsSoA *hits_layers,
  const int n_hits ) {
  for ( int i_hit = 0; i_hit < n_hits; ++i_hit ) {
    const VeloUTTracking::Hit* hit = hits[i_hit];
    const int planeCode = hit->planeCode();
    const float ui = x_pos_layers[ planeCode ][ hitCandidateIndices[i_hit] ];
    const float ci = hit->cosT();
    const int layer_offset = hits_layers->layer_offset[ planeCode ];
    const int hit_index = hitCandidatesInLayers[planeCode][ hitCandidateIndices[i_hit] ];
    const float z  = hits_layers->zAtYEq0( layer_offset + hit_index );
    const float dz = 0.001*(z - PrVeloUTConst::zMidUT);
    const float wi = hit->weight();

    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }
}

__host__ __device__ void addChi2s(
  const float xUTFit,
  const float xSlopeUTFit,
  float& chi2 ,
  const VeloUTTracking::Hit** hits,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  const int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  VeloUTTracking::HitsSoA *hits_layers,
  const int n_hits ) {
  for ( int i_hit = 0; i_hit < n_hits; ++i_hit ) {
    const VeloUTTracking::Hit* hit = hits[i_hit];
    const int planeCode = hit->planeCode();
    const int layer_offset = hits_layers->layer_offset[ planeCode ];
    const int hit_index = hitCandidatesInLayers[planeCode][ hitCandidateIndices[i_hit] ];
    const float zd = hits_layers->zAtYEq0( layer_offset + hit_index );
    const float xd = xUTFit + xSlopeUTFit*(zd-PrVeloUTConst::zMidUT);
    const float x  = x_pos_layers[ planeCode ][ hitCandidateIndices[i_hit] ];
    const float du = xd - x;
    chi2 += (du*du)*hit->weight();

  }
}


