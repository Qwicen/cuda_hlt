#include "PrVeloUT.cuh"

//-----------------------------------------------------------------------------
// Implementation file for PrVeloUT
//
// 2007-05-08: Mariusz Witek
// 2017-03-01: Christoph Hasse (adapt to future framework)
// 2018-05-05: Plácido Fernández (make standalone)
// 2018-07:    Dorothea vom Bruch (convert to C and then CUDA code)
//-----------------------------------------------------------------------------

//=============================================================================
// Reject tracks outside of acceptance or pointing to the beam pipe
//=============================================================================
__host__ __device__ bool veloTrackInUTAcceptance(
  const MiniState& state
) {
  const float xMidUT = state.x + state.tx*( PrVeloUTConst::zMidUT - state.z);
  const float yMidUT = state.y + state.ty*( PrVeloUTConst::zMidUT - state.z);

  if( xMidUT*xMidUT+yMidUT*yMidUT  < PrVeloUTConst::centralHoleSize*PrVeloUTConst::centralHoleSize ) return false;
  if( (std::abs(state.tx) > PrVeloUTConst::maxXSlope) || (std::abs(state.ty) > PrVeloUTConst::maxYSlope) ) return false;

  if(PrVeloUTConst::passTracks && std::abs(xMidUT) < PrVeloUTConst::passHoleSize && std::abs(yMidUT) < PrVeloUTConst::passHoleSize) {
    return false;
  }

  return true;
}

// get the low pos for a window
__host__ __device__ int win_lpos(
  const int i_track,
  const int layer) 
{
  return (i_track * VeloUTTracking::n_layers * 2) + (layer * 2);
};

// get the high pos for a window
__host__ __device__ int win_hpos(
  const int i_track,
  const int layer) 
{
  // return (i_track * VeloUTTracking::n_layers * 2) + (layer * 2) + 1;
  return win_lpos(i_track, layer) + 1;
};

//=============================================================================
// Binary search a hit in the range
//=============================================================================
__host__ __device__ void binary_search_range(
  const int layer,
  const UTHits& ut_hits,
  const UTHitCount& ut_hit_count,
  const float ut_dxDy,
  const float low_bound_x,
  const float up_bound_x,
  const float xTolNormFact,
  const float yApprox,
  const float xOnTrackProto,
  const int layer_offset,
  int& high_hit_pos,
  int& low_hit_pos)
{
  const int num_hits_layer = ut_hit_count.n_hits_layers[layer];

  int min = layer_offset; // first hit of the layer
  int max = layer_offset + (num_hits_layer - 1); // last hit of the layer
  int guess = 0;

  // const float xx = ut_hits.xAt(layer_offset + guess, yApprox, ut_dxDy);
  // const float dx = xx - xOnTrackProto;

  high_hit_pos = -1;
  low_hit_pos = -1;

  // look for the low limit
  guess = (max + min) / 2;
  while (max != min+1) {
    const float xx = ut_hits.xAt(layer_offset + guess, yApprox, ut_dxDy);
    const float dx = xx - xOnTrackProto;
    if (dx < -xTolNormFact) {
      min = guess;
    } else {
      max = guess;
    }   
    guess = (max + min) / 2;
  }
  low_hit_pos = guess;

  // look for the high limit
  min = low_hit_pos;
  max = layer_offset + (num_hits_layer - 1); // last hit of the layer
  guess = (max + min) / 2;
  while (max != min+1) {
    const float xx = ut_hits.xAt(layer_offset + guess, yApprox, ut_dxDy);
    const float dx = xx - xOnTrackProto;
    if (dx < xTolNormFact) {
      max = guess;
    } else {
      min = guess;
    }   
    guess = (max + min) / 2;
  }
  high_hit_pos = guess;

  // float lh = ut_hits.xAtYEq0[layer_offset + low_hit_pos]; 
  // float hh = ut_hits.xAtYEq0[layer_offset + high_hit_pos];
  // printf("low_hit_pos: %d, high_hit_pos: %d, lh: %f, hh: %f, xTolNormFact: %f\n", low_hit_pos, high_hit_pos, lh, hh, xTolNormFact);
}

//=============================================================================
// Get the windows
//=============================================================================
__host__ __device__ void get_windows(
  const int i_track,
  const MiniState& veloState,
  const float* fudgeFactors,
  const UTHits& ut_hits,
  const UTHitCount& ut_hit_count,
  const float* ut_dxDy,
  int* windows_layers) 
{
  // -- This is hardcoded, so faster
  // -- If you ever change the Table in the magnet tool, this will be wrong
  const float absSlopeY = std::abs( veloState.ty );
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
  const float invTheta = std::min(500., 1.0/std::sqrt(veloState.tx*veloState.tx+veloState.ty*veloState.ty));
  const float minMom   = std::max(PrVeloUTConst::minPT*invTheta, float(1.5)*Gaudi::Units::GeV);
  const float xTol     = std::abs(1. / ( PrVeloUTConst::distToMomentum * minMom ));
  const float yTol     = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * xTol;

  const float dxDyHelper[N_LAYERS] = {0., 1., -1., 0};

  for (int layer=0; layer<N_LAYERS; ++layer) {

    int layer_offset = ut_hit_count.layer_offsets[layer];

    const float dxDy   = ut_dxDy[layer];
    const float zLayer = ut_hits.zAtYEq0[layer_offset]; 

    const float yAtZ   = veloState.y + veloState.ty*(zLayer - veloState.z);
    const float xLayer = veloState.x + veloState.tx*(zLayer - veloState.z);
    const float yLayer = yAtZ + yTol * dxDyHelper[layer];

    // const float normFactNum = normFact[layer];
    const float invNormFact = 1.0/normFact[layer];

    const float lowerBoundX =
      (xLayer - dxDy*yLayer) - xTol*invNormFact - std::abs(veloState.tx)*PrVeloUTConst::intraLayerDist;
    const float upperBoundX =
      (xLayer - dxDy*yLayer) + xTol*invNormFact + std::abs(veloState.tx)*PrVeloUTConst::intraLayerDist;

    const float zInit = ut_hits.zAtYEq0[layer_offset];
    const float xTolNormFact = xTol*invNormFact;
    const float yApprox = veloState.y + veloState.ty * (zInit - veloState.z);
    const float xOnTrackProto = veloState.x + veloState.tx*(zInit - veloState.z);

    binary_search_range(
      layer,
      ut_hits,
      ut_hit_count,
      ut_dxDy[layer],
      lowerBoundX,
      upperBoundX,
      xTolNormFact,
      yApprox,
      xOnTrackProto,
      layer_offset,
      windows_layers[win_lpos(i_track, layer)],
      windows_layers[win_hpos(i_track, layer)]);
  }
}

//=============================================================================
// Find the hits in all layers
//=============================================================================
__host__ __device__ bool getHits(
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int posLayers[4][85],
  const UTHits& ut_hits,
  const UTHitCount& ut_hit_count,
  const float* fudgeFactors, 
  const MiniState& trState,
  const float* ut_dxDy)
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
  const float invTheta = std::min(500., 1.0/std::sqrt(trState.tx*trState.tx+trState.ty*trState.ty));
  //const float minMom   = std::max(PrVeloUTConst::minPT*invTheta, PrVeloUTConst::minMomentum);
  const float minMom   = std::max(PrVeloUTConst::minPT*invTheta, float(1.5)*Gaudi::Units::GeV);
  const float xTol     = std::abs(1. / ( PrVeloUTConst::distToMomentum * minMom ));
  const float yTol     = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * xTol;

  int nLayers = 0;

  const float dxDyHelper[VeloUTTracking::n_layers] = {0., 1., -1., 0};
  
  for (int layer=0; layer<VeloUTTracking::n_layers; ++layer) {

    if( (layer == 3 || layer == 4) && nLayers == 0) return false;
    if( layer == 4 && nLayers < 2 ) return false;
    if( ut_hit_count.n_hits_layers[layer] == 0 ) continue;

    int layer_offset = ut_hit_count.layer_offsets[layer];

    const float dxDy   = ut_dxDy[layer];
    const float zLayer = ut_hits.zAtYEq0[layer_offset + 0]; 

    const float yAtZ   = trState.y + trState.ty*(zLayer - trState.z);
    const float xLayer = trState.x + trState.tx*(zLayer - trState.z);
    const float yLayer = yAtZ + yTol * dxDyHelper[layer];

    const float normFactNum = normFact[layer];
    const float invNormFact = 1.0/normFactNum;

    const float lowerBoundX =
      (xLayer - dxDy*yLayer) - xTol*invNormFact - std::abs(trState.tx)*PrVeloUTConst::intraLayerDist;
    const float upperBoundX =
      (xLayer - dxDy*yLayer) + xTol*invNormFact + std::abs(trState.tx)*PrVeloUTConst::intraLayerDist;

    const int indexLowProto = 
      lowerBoundX > 0 ? std::sqrt( std::abs(lowerBoundX)*2.0 ) + 42 : 42 - std::sqrt( std::abs(lowerBoundX)*2.0 );
    const int indexHiProto  = 
      upperBoundX > 0 ? std::sqrt( std::abs(upperBoundX)*2.0 ) + 43 : 43 - std::sqrt( std::abs(upperBoundX)*2.0 );

    const int indexLow  = std::max( indexLowProto, 0 );
    const int indexHi   = std::min( indexHiProto, 84);

    size_t posBeg = posLayers[layer][ indexLow ];
    size_t posEnd = posLayers[layer][ indexHi  ];

    while ( (ut_hits.xAtYEq0[layer_offset + posBeg] < lowerBoundX) && (posBeg != ut_hit_count.n_hits_layers[layer] ) ) {
      ++posBeg;
    }
      
    if (posBeg == ut_hit_count.n_hits_layers[layer]) continue;

    findHits(posBeg, posEnd,
      ut_hits, layer_offset, ut_dxDy[layer],
      trState, xTol*invNormFact, invNormFact,
      hitCandidatesInLayers[layer], n_hitCandidatesInLayers[layer],
      x_pos_layers[layer]);

    nLayers += int( !( n_hitCandidatesInLayers[layer] == 0 ) );    
  }

  return nLayers > 2;
}

//=========================================================================
// hits_to_track
//=========================================================================
__host__ __device__ bool find_best_hits(
  const int i_track,
  const int* windows_layers,
  const UTHits& ut_hits,
  const UTHitCount& ut_hit_count,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const bool forward,
  TrackHelper& helper,
  float* x_hit_layer,
  int* bestHitCandidateIndices)
{
  // handle forward / backward cluster search
  int layers[N_LAYERS];
  for ( int i_layer = 0; i_layer < N_LAYERS; ++i_layer ) {
      if ( forward ) layers[i_layer] = i_layer;
      else layers[i_layer] = N_LAYERS - 1 - i_layer;
  }

  // Go through the layers
  bool fourLayerSolution = false;
  int hitCandidateIndices[N_LAYERS];

  // Get the needed stuff
  const float yyProto = velo_state.y - velo_state.ty*velo_state.z;

  // Get windows of layers
  const int from0 = windows_layers[win_lpos(i_track, layers[0])];
  const int to0 =   windows_layers[win_hpos(i_track, layers[0])];
  const int from2 = windows_layers[win_lpos(i_track, layers[2])];
  const int to2 =   windows_layers[win_hpos(i_track, layers[2])];
  const int from1 = windows_layers[win_lpos(i_track, layers[1])];
  const int to1 =   windows_layers[win_hpos(i_track, layers[1])];
  const int from3 = windows_layers[win_lpos(i_track, layers[3])];
  const int to3 =   windows_layers[win_hpos(i_track, layers[3])];

  for ( int i_hit0 = from0; i_hit0 < to0; ++i_hit0 ) {

    // x_pos_layers calc
    // TODO put the .isNotYCompatible check
    // TODO do that for all the layers
    const float yy0 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit0]);
    x_hit_layer[0] = ut_hits.xAt(i_hit0, yy0, ut_dxDy[layers[0]]);
    // ---------------------------------------

    const float zhitLayer0 = ut_hits.zAtYEq0[i_hit0];
    hitCandidateIndices[0] = i_hit0;

    for ( int i_hit2 = from2; i_hit2 < to2; ++i_hit2 ) {
      // x_pos_layers calc
      const float yy2 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit2]);
      x_hit_layer[2] = ut_hits.xAt(i_hit2, yy2, ut_dxDy[layers[2]]);
      // ---------------------------------------

      const float zhitLayer2  = ut_hits.zAtYEq0[i_hit2];
      hitCandidateIndices[2] = i_hit2;
      
      const float tx = (x_hit_layer[2] - x_hit_layer[0])/(zhitLayer2 - zhitLayer0);
      if( std::abs(tx-velo_state.tx) > PrVeloUTConst::deltaTx2 ) continue;
            
      float hitTol = PrVeloUTConst::hitTol2;
      int index_best_hit_1 = -1;

      for ( int i_hit1 = from1; i_hit1 < to1; ++i_hit1 ) {
        // x_pos_layers calc
        const float yy1 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit1]);
        x_hit_layer[1] = ut_hits.xAt(i_hit1, yy1, ut_dxDy[layers[1]]);
        // ---------------------------------------        
        const float zhitLayer1  = ut_hits.zAtYEq0[i_hit1];
       
        const float xextrapLayer1 = x_hit_layer[0] + tx*(zhitLayer1-zhitLayer0);
        if(std::abs(x_hit_layer[1] - xextrapLayer1) < hitTol){
          hitTol = std::abs(x_hit_layer[1] - xextrapLayer1);
          index_best_hit_1 = i_hit1;
          hitCandidateIndices[1] = i_hit1;
        }
      } // loop over layer 1
      
      if( fourLayerSolution && index_best_hit_1 < 0 ) continue;

      int index_best_hit_3 = -1;
      hitTol = PrVeloUTConst::hitTol2;

      for ( int i_hit3 = from3; i_hit3 < to3; ++i_hit3 ) {
        // x_pos_layers calc
        const float yy3 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit3]);
        x_hit_layer[3] = ut_hits.xAt(i_hit3, yy3, ut_dxDy[layers[3]]);
        // ---------------------------------------      
        const float zhitLayer3  = ut_hits.zAtYEq0[i_hit3];
        
        const float xextrapLayer3 = x_hit_layer[2] + tx*(zhitLayer3-zhitLayer2);
        if(std::abs(x_hit_layer[3] - xextrapLayer3) < hitTol){
          hitTol = std::abs(x_hit_layer[3] - xextrapLayer3);
          index_best_hit_3 = i_hit3;
          hitCandidateIndices[3] = i_hit3;
        }
      }
     
      // -- All hits found
      if ( index_best_hit_1 > 0 && index_best_hit_3 > 0 ) {
        const int hitIndices[4] = {i_hit0, index_best_hit_1, i_hit2, index_best_hit_3};
        simpleFit<4>(
          x_hit_layer,
          hitCandidateIndices,
          ut_hits, 
          hitIndices, 
          velo_state, 
          ut_dxDy,
          bestHitCandidateIndices,
          helper);
        
        if(!fourLayerSolution && helper.n_hits > 0){
          fourLayerSolution = true;
        }
        continue;
      }

      // -- Nothing found in layer 3
      if( !fourLayerSolution && index_best_hit_1 > 0 ){
        const int hitIndices[3] = {i_hit0, index_best_hit_1, i_hit2};
        simpleFit<3>(
          x_hit_layer,
          hitCandidateIndices, 
          ut_hits, 
          hitIndices, 
          velo_state, 
          ut_dxDy,
          bestHitCandidateIndices,
          helper);
        continue;
      }
      // -- Nothing found in layer 1
      if( !fourLayerSolution && x_hit_layer[3] > 0 ){
        hitCandidateIndices[1] = hitCandidateIndices[3];  // hit3 saved in second position of hits4fit
        const int hitIndices[3] = {i_hit0, index_best_hit_3, i_hit2};
        simpleFit<3>(
          x_hit_layer,
          hitCandidateIndices,  
          ut_hits, 
          hitIndices, 
          velo_state, 
          ut_dxDy,
          bestHitCandidateIndices,
          helper);
        continue;
      }
    }
  }

  return fourLayerSolution;
}

// -- These things are all hardcopied from the PrTableForFunction
// -- and PrUTMagnetTool
// -- If the granularity or whatever changes, this will give wrong results
// TODO put this as a lambda
__host__ __device__ int masterIndex(const int index1, const int index2, const int index3){
  return (index3*11 + index2)*31 + index1;
}

//=========================================================================
// Create the Velo-UT tracks
//=========================================================================
// TODO put all the "magic" numbers to meaningful constants
__host__ __device__ void prepareOutputTrack(
  const int i_track,
  const Velo::Consolidated::Hits& velo_track_hits,
  const uint velo_track_hit_number,
  const TrackHelper& helper,
  const MiniState& velo_state,
  const int* windows_layers,
  const UTHits& ut_hits,
  const UTHitCount& ut_hit_count,
  const float* x_hit_layer,
  const int* hitCandidateIndices,
  const float* bdlTable,
  VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks],
  int* n_veloUT_tracks) 
{
  //== Handle states. copy Velo one, add UT.
  const float zOrigin = (std::fabs(velo_state.ty) > 0.001)
    ? velo_state.z - velo_state.y / velo_state.ty
    : velo_state.z - velo_state.x / velo_state.tx;

  // -- These are calculations, copied and simplified from PrTableForFunction
  const std::array<float,3> var = { velo_state.ty, zOrigin, velo_state.z };

  const int index1 = std::max(0, std::min( 30, int((var[0] + 0.3)/0.6*30) ));
  const int index2 = std::max(0, std::min( 10, int((var[1] + 250)/500*10) ));
  const int index3 = std::max(0, std::min( 10, int( var[2]/800*10)        ));

  assert( masterIndex(index1, index2, index3) < PrUTMagnetTool::N_bdl_vals );
  float bdl = bdlTable[masterIndex(index1, index2, index3)];

  const float bdls[3] = { bdlTable[masterIndex(index1+1, index2,index3)],
                          bdlTable[masterIndex(index1,index2+1,index3)],
                          bdlTable[masterIndex(index1,index2,index3+1)] };
  const float deltaBdl[3]   = { 0.02, 50.0, 80.0 };
  const float boundaries[3] = { -0.3f + float(index1)*deltaBdl[0],
                                -250.0f + float(index2)*deltaBdl[1],
                                0.0f + float(index3)*deltaBdl[2] };

  // -- This is an interpolation, to get a bit more precision
  float addBdlVal = 0.0;
  const float minValsBdl[3] = { -0.3, -250.0, 0.0 };
  const float maxValsBdl[3] = { 0.3, 250.0, 800.0 };
  for(int i=0; i<3; ++i) {
    if( var[i] < minValsBdl[i] || var[i] > maxValsBdl[i] ) continue;
    const float dTab_dVar =  (bdls[i] - bdl) / deltaBdl[i];
    const float dVar = (var[i]-boundaries[i]);
    addBdlVal += dTab_dVar*dVar;
  }
  bdl += addBdlVal;
  // ----

  const float qpxz2p =-1*std::sqrt(1.+velo_state.ty*velo_state.ty)/bdl*3.3356/Gaudi::Units::GeV;
  const float qop = (std::abs(bdl) < 1.e-8) ? 0.0 : helper.bestParams[0]*qpxz2p;

  // -- Don't make tracks that have grossly too low momentum
  // -- Beware of the momentum resolution!
  const float p  = 1.3*std::abs(1/qop);
  const float pt = p*std::sqrt(velo_state.tx*velo_state.tx + velo_state.ty*velo_state.ty);

  if( p < PrVeloUTConst::minMomentum || pt < PrVeloUTConst::minPT ) return;

#ifdef __CUDA_ARCH__
  uint n_tracks = atomicAdd(n_veloUT_tracks, 1);
#else
  (*n_veloUT_tracks)++;
  uint n_tracks = *n_veloUT_tracks - 1;
#endif

  const float txUT = helper.bestParams[3];

  // TODO: Maybe have a look and optimize this if possible
  VeloUTTracking::TrackUT track;
  track.hitsNum = 0;
  for (int i=0; i<velo_track_hit_number; ++i) {
    track.addLHCbID(velo_track_hits.LHCbID[i]);
    assert( track.hitsNum < VeloUTTracking::max_track_size);
  }
  track.set_qop( qop );
  
  // Adding overlap hits
  for ( int i_hit = 0; i_hit < helper.n_hits; ++i_hit ) {
    const int hit_index = helper.bestHitIndices[i_hit];
    
    track.addLHCbID( ut_hits.LHCbID[hit_index] );
    assert( track.hitsNum < VeloUTTracking::max_track_size);
    
    const int planeCode = ut_hits.planeCode[hit_index];
    const float xhit = x_hit_layer[ planeCode ];
    // const float xhit = x_pos_layers[ planeCode ][ hitCandidateIndices[i_hit] ];
    const float zhit = ut_hits.zAtYEq0[hit_index];

    const int layer_offset = ut_hit_count.layer_offsets[ planeCode ];
    // Search in the window

    // Get windows of layers
    const int from = windows_layers[win_lpos(i_track, planeCode)];
    const int to =   windows_layers[win_hpos(i_track, planeCode)];
    for ( int i_hit = from; i_hit < to; ++i_hit ) {
      const float zohit  = ut_hits.zAtYEq0[i_hit];
      if(zohit==zhit) continue;

      const float xohit = x_hit_layer[ planeCode ];
      // const float xohit = x_pos_layers[ planeCode ][ i_ohit];
      const float xextrap = xhit + txUT*(zhit-zohit);
      if( xohit-xextrap < -PrVeloUTConst::overlapTol) continue;
      if( xohit-xextrap > PrVeloUTConst::overlapTol) break;
      
      track.addLHCbID(ut_hits.LHCbID[i_hit]);
      assert( track.hitsNum < VeloUTTracking::max_track_size);
      
      // -- only one overlap hit
      break;
    }

    // for ( int i_ohit = 0; i_ohit < n_hitCandidatesInLayers[planeCode]; ++i_ohit ) {
    //   const int ohit_index = hitCandidatesInLayers[planeCode][i_ohit];
    //   const float zohit  = ut_hits.zAtYEq0[layer_offset + ohit_index];
      
    //   if(zohit==zhit) continue;
      
    //   const float xohit = x_hit_layers[ planeCode ];
    //   // const float xohit = x_pos_layers[ planeCode ][ i_ohit];
    //   const float xextrap = xhit + txUT*(zhit-zohit);
    //   if( xohit-xextrap < -PrVeloUTConst::overlapTol) continue;
    //   if( xohit-xextrap > PrVeloUTConst::overlapTol) break;
      
    //   track.addLHCbID(ut_hits.LHCbID[layer_offset + ohit_index]);
    //   assert( track.hitsNum < VeloUTTracking::max_track_size);
      
    //   // -- only one overlap hit
    //   break;
    // }
  }
  assert( n_tracks < VeloUTTracking::max_num_tracks );
  VeloUT_tracks[n_tracks] = track;

  /*
  outTr.x = velo_state.x;
  outTr.y = velo_state.y;
  outTr.z = velo_state.z;
  outTr.tx = velo_state.tx;
  outTr.ty = velo_state.ty;
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

//==============================================================================
// -- Method to cache some starting points for the search
// -- This is actually faster than binary searching the full array
// -- Granularity hardcoded for the moment.
// -- Idea is: UTb has dimensions in x (at y = 0) of about -860mm -> 860mm
// -- The indices go from 0 -> 84, and shift by -42, leading to -42 -> 42
// -- Taking the higher density of hits in the center into account, the positions of the iterators are
// -- calculated as index*index/2, where index = [ -42, 42 ], leading to
// -- -882mm -> 882mm
// -- The last element is an "end" iterator, to make sure we never go out of bound
//==============================================================================
__host__ __device__ void fillIterators(
  UTHits& ut_hits,
  UTHitCount& ut_hit_count,
  int posLayers[4][85] )
{
    
  for(int iStation = 0; iStation < 2; ++iStation){
    for(int iLayer = 0; iLayer < 2; ++iLayer){
      int layer = 2*iStation + iLayer;
      int layer_offset = ut_hit_count.layer_offsets[layer];
      uint n_hits_layer = ut_hit_count.n_hits_layers[layer];
      
      size_t pos = 0;
      // to do: check whether there is an efficient thrust implementation for this
      fillArray( posLayers[layer], 85, pos );
      
      int bound = -42.0;
      // to do : make copysignf
      float val = std::copysign(float(bound*bound)/2.0, bound);
      
      // TODO add bounds checking
      for ( ; pos != n_hits_layer; ++pos) {
        while( ut_hits.xAtYEq0[layer_offset + pos] > val){
          posLayers[layer][bound+42] = pos;
          ++bound;
          val = std::copysign(float(bound*bound)/2.0, bound);
        }
      }
      
      fillArrayAt(
        posLayers[layer],
        42 + bound,
        85 - 42 - bound,
        n_hits_layer
      );
    }
  }
}


//==============================================================================
// Finds the hits in a given layer within a certain range
//==============================================================================
__host__ __device__ void findHits( 
  const size_t posBeg,
  const size_t posEnd,
  const UTHits& ut_hits,
  const uint layer_offset,
  const float ut_dxDy,
  const MiniState& myState, 
  const float xTolNormFact,
  const float invNormFact,
  int hitCandidatesInLayer[VeloUTTracking::max_hit_candidates_per_layer],
  int &n_hitCandidatesInLayer,
  float x_pos_layers[VeloUTTracking::max_hit_candidates_per_layer])
{
  const float zInit = ut_hits.zAtYEq0[layer_offset + posBeg];
  const float yApprox = myState.y + myState.ty * (zInit - myState.z);
  
  size_t pos = posBeg;
  while ( 
   pos <= posEnd && 
   ut_hits.isNotYCompatible( layer_offset + pos, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(xTolNormFact) )
   ) { ++pos; }

  const float xOnTrackProto = myState.x + myState.tx*(zInit - myState.z);
  const float yyProto =       myState.y - myState.ty*myState.z;
  
  for (int i=pos; i<posEnd; ++i) {
    const float xx = ut_hits.xAt(layer_offset + i, yApprox, ut_dxDy); 
    const float dx = xx - xOnTrackProto;
    
    if( dx < -xTolNormFact ) continue;
    if( dx >  xTolNormFact ) break; 
    
    // -- Now refine the tolerance in Y
    if ( ut_hits.isNotYCompatible( layer_offset + i, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx*invNormFact)) ) continue;
    
    const float zz = ut_hits.zAtYEq0[layer_offset + i]; 
    const float yy = yyProto +  myState.ty*zz;
    const float xx2 = ut_hits.xAt(layer_offset + i, yy, ut_dxDy);
        
    hitCandidatesInLayer[n_hitCandidatesInLayer] = i;
    x_pos_layers[n_hitCandidatesInLayer] = xx2;
    
    n_hitCandidatesInLayer++;

    if ( n_hitCandidatesInLayer >= VeloUTTracking::max_hit_candidates_per_layer )
      printf("%u > %u !! \n", n_hitCandidatesInLayer, VeloUTTracking::max_hit_candidates_per_layer);
    assert( n_hitCandidatesInLayer < VeloUTTracking::max_hit_candidates_per_layer );
  }
  for ( int i_hit = 0; i_hit < n_hitCandidatesInLayer; ++i_hit ) {
    assert( hitCandidatesInLayer[i_hit] < VeloUTTracking::max_numhits_per_event );
  }
}
