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


//=====================================================================
// Propagate to end of Velo z position (z=770mm)
// only propagate x, y, z; covariance matrix is not needed
//=====================================================================
__host__ __device__ void propagate_state_to_end_velo(
  Velo::State& velo_state
) {
  const float dz = VeloTracking::z_endVelo - velo_state.z;
  velo_state.x += dz * velo_state.tx;
  velo_state.y += dz * velo_state.ty;
  velo_state.z = VeloTracking::z_endVelo;
}


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

//=============================================================================
// Find the hits
//=============================================================================
__device__ bool getHits(
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  UTHits& ut_hits,
  UTHitOffsets& ut_hit_offsets,
  const float* fudgeFactors, 
  const MiniState& trState,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets)
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

  float dxDyHelper[VeloUTTracking::n_layers] = {0., 1., -1., 0};
  for(int iStation = 0; iStation < 2; ++iStation) {

    if( iStation == 1 && nLayers == 0 ) return false;

    for(int iLayer = 0; iLayer < 2; ++iLayer) {
      if( iStation == 1 && iLayer == 1 && nLayers < 2 ) return false;

      int layer = 2*iStation+iLayer;
      int layer_offset = ut_hit_offsets.layer_offset(layer);
      
      if( ut_hit_offsets.layer_number_of_hits(layer) == 0 ) continue;
      const float dxDy   = ut_dxDy[layer];
      const float zLayer = ut_hits.zAtYEq0[layer_offset + 0]; 

      const float yAtZ   = trState.y + trState.ty*(zLayer - trState.z);
      const float xLayer = trState.x + trState.tx*(zLayer - trState.z);
      const float yLayer = yAtZ + yTol * dxDyHelper[layer];

      const float normFactNum = normFact[2*iStation + iLayer];
      const float invNormFact = 1.0/normFactNum;

      const float lowerBoundX =
        (xLayer - dxDy*yLayer) - xTol*invNormFact - std::abs(trState.tx)*PrVeloUTConst::intraLayerDist;
      const float upperBoundX =
        (xLayer - dxDy*yLayer) + xTol*invNormFact + std::abs(trState.tx)*PrVeloUTConst::intraLayerDist;

      // Find sector group for lowerBoundX and upperBoundX
      const uint first_sector_group_in_layer = dev_unique_x_sector_layer_offsets[layer];
      const uint last_sector_group_in_layer = dev_unique_x_sector_layer_offsets[layer+1];
      const uint number_of_sector_groups = last_sector_group_in_layer - first_sector_group_in_layer;

      uint lowerBoundSectorGroup = first_sector_group_in_layer;
      uint upperBoundSectorGroup = last_sector_group_in_layer - 1;

      // The window of search is out of bounds
      if (upperBoundX < dev_unique_sector_xs[first_sector_group_in_layer] ||
          lowerBoundX > dev_unique_sector_xs[last_sector_group_in_layer - 1]) {
        continue;
      }

      for (int i=first_sector_group_in_layer + 2; i<last_sector_group_in_layer; ++i) {
        if (dev_unique_sector_xs[i] > lowerBoundX) {
          lowerBoundSectorGroup = i-2;
          break;
        }
      }

      for (int i=0; i<number_of_sector_groups; ++i) {
        const uint current_sector_group = first_sector_group_in_layer + i;
        if (dev_unique_sector_xs[current_sector_group] > upperBoundX) {
          if (i == number_of_sector_groups - 1) {
            upperBoundSectorGroup = current_sector_group;
          } else {
            upperBoundSectorGroup = current_sector_group + 1;
          }
          break;
        }
      }

      assert(upperBoundSectorGroup < last_sector_group_in_layer);
      assert(lowerBoundSectorGroup >= first_sector_group_in_layer);
      assert(lowerBoundSectorGroup < upperBoundSectorGroup);

      findHits(lowerBoundSectorGroup, upperBoundSectorGroup,
        ut_hits, ut_hit_offsets, layer_offset, layer, ut_dxDy,
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
  UTHits& ut_hits,
  UTHitOffsets& ut_hit_offsets,
  TrackHelper& helper,
  MiniState& state,
  const float* ut_dxDy,
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

    const int layer_offset0 = ut_hit_offsets.layer_offset(layers[0]);
    const int hit_index0    = layer_offset0 + hitCandidatesInLayers[ layers[0] ][i_hit0];
    const float xhitLayer0  = x_pos_layers[layers[0]][i_hit0];
    const float zhitLayer0  = ut_hits.zAtYEq0[hit_index0];
    hitCandidateIndices[0] = i_hit0;
    
    for ( int i_hit2 = 0; i_hit2 < n_hitCandidatesInLayers[ layers[2] ]; ++i_hit2 ) {

      const int layer_offset2 = ut_hit_offsets.layer_offset(layers[2]);
      const int hit_index2    = layer_offset2 + hitCandidatesInLayers[ layers[2] ][i_hit2];
      const float xhitLayer2  = x_pos_layers[layers[2]][i_hit2];
      const float zhitLayer2  = ut_hits.zAtYEq0[hit_index2];
      hitCandidateIndices[2] = i_hit2;
      
      const float tx = (xhitLayer2 - xhitLayer0)/(zhitLayer2 - zhitLayer0);
      if( std::abs(tx-state.tx) > PrVeloUTConst::deltaTx2 ) continue;
            
      int IndexBestHit1 = -10;
      float hitTol = PrVeloUTConst::hitTol2;
      for ( int i_hit1 = 0; i_hit1 < n_hitCandidatesInLayers[ layers[1] ]; ++i_hit1 ) {

        const int layer_offset1 = ut_hit_offsets.layer_offset(layers[1]);
        const int hit_index1    = layer_offset1 + hitCandidatesInLayers[ layers[1] ][i_hit1];
        const float xhitLayer1  = x_pos_layers[layers[1]][i_hit1];
        const float zhitLayer1  = ut_hits.zAtYEq0[hit_index1];
       
        const float xextrapLayer1 = xhitLayer0 + tx*(zhitLayer1-zhitLayer0);
        if(std::abs(xhitLayer1 - xextrapLayer1) < hitTol){
          hitTol = std::abs(xhitLayer1 - xextrapLayer1);
          IndexBestHit1 = hit_index1;
          hitCandidateIndices[1] = i_hit1;
        }
      } // loop over layer 1
      
      if( fourLayerSolution && IndexBestHit1 < 0 ) continue;

      int IndexBestHit3 = -10;
      hitTol = PrVeloUTConst::hitTol2;
      for ( int i_hit3 = 0; i_hit3 < n_hitCandidatesInLayers[ layers[3] ]; ++i_hit3 ) {

        const int layer_offset3 = ut_hit_offsets.layer_offset(layers[3]);
        const int hit_index3    = layer_offset3 + hitCandidatesInLayers[ layers[3] ][i_hit3];
        const float xhitLayer3  = x_pos_layers[layers[3]][i_hit3];
        const float zhitLayer3  = ut_hits.zAtYEq0[hit_index3];
        
        const float xextrapLayer3 = xhitLayer2 + tx*(zhitLayer3-zhitLayer2);
        if(std::abs(xhitLayer3 - xextrapLayer3) < hitTol){
          hitTol = std::abs(xhitLayer3 - xextrapLayer3);
          IndexBestHit3 = hit_index3;
          hitCandidateIndices[3] = i_hit3;
        }
      } // loop over layer 3
     
      // -- All hits found
      if ( IndexBestHit1 > 0 && IndexBestHit3 > 0 ) {
        const int hitIndices[4] = {hit_index0, IndexBestHit1, hit_index2, IndexBestHit3};
        simpleFit<4>(x_pos_layers, hitCandidateIndices, bestHitCandidateIndices, hitCandidatesInLayers, ut_hits, hitIndices, helper, state, ut_dxDy);
        
        if(!fourLayerSolution && helper.n_hits > 0){
          fourLayerSolution = true;
        }
        continue;
      }

      // -- Nothing found in layer 3
      if( !fourLayerSolution && IndexBestHit1 > 0 ){
        const int hitIndices[3] = {hit_index0, IndexBestHit1, hit_index2};
        simpleFit<3>(x_pos_layers, hitCandidateIndices, bestHitCandidateIndices, hitCandidatesInLayers, ut_hits, hitIndices, helper, state, ut_dxDy);
        continue;
      }
      // -- Nothing found in layer 1
      if( !fourLayerSolution && IndexBestHit3 > 0 ){
        hitCandidateIndices[1] = hitCandidateIndices[3];  // hit3 saved in second position of hits4fit
        const int hitIndices[3] = {hit_index0, IndexBestHit3, hit_index2};
        simpleFit<3>(x_pos_layers, hitCandidateIndices, bestHitCandidateIndices, hitCandidatesInLayers, ut_hits, hitIndices, helper, state, ut_dxDy);
        continue;
      }
      
    }
  }

  return fourLayerSolution;
}
//=========================================================================
// Create the Velo-UT tracks
//=========================================================================
__host__ __device__ void prepareOutputTrack(
  const Velo::Consolidated::Hits& velo_track_hits,
  const uint velo_track_hit_number,
  const TrackHelper& helper,
  const MiniState& state,
  int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  int n_hitCandidatesInLayers[VeloUTTracking::n_layers],
  UTHits& ut_hits,
  UTHitOffsets& ut_hit_offsets,
  const float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer],
  const int hitCandidateIndices[VeloUTTracking::n_layers],
  VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks],
  int* n_veloUT_tracks,
  const int i_velo_track,
  const float* bdlTable) {

  //== Handle states. copy Velo one, add UT.
  const float zOrigin = (std::fabs(state.ty) > 0.001)
    ? state.z - state.y / state.ty
    : state.z - state.x / state.tx;

  // -- These are calculations, copied and simplified from PrTableForFunction
  const std::array<float,3> var = { state.ty, zOrigin, state.z };

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

  const float qpxz2p =-1*std::sqrt(1.+state.ty*state.ty)/bdl*3.3356/Gaudi::Units::GeV;
  const float qop = (std::abs(bdl) < 1.e-8) ? 0.0 : helper.bestParams[0]*qpxz2p;

  // -- Don't make tracks that have grossly too low momentum
  // -- Beware of the momentum resolution!
  const float p  = 1.3*std::abs(1/qop);
  const float pt = p*std::sqrt(state.tx*state.tx + state.ty*state.ty);

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
  track.veloTrackIndex = i_velo_track;
  
  // Adding overlap hits
  for ( int i_hit = 0; i_hit < helper.n_hits; ++i_hit ) {
    const int hit_index = helper.bestHitIndices[i_hit];
    
    track.addLHCbID( ut_hits.LHCbID[hit_index] );
    assert( track.hitsNum < VeloUTTracking::max_track_size);
    
    const int planeCode = ut_hits.planeCode[hit_index];
    const float xhit = x_pos_layers[ planeCode ][ hitCandidateIndices[i_hit] ];
    const float zhit = ut_hits.zAtYEq0[hit_index];

    const int layer_offset = ut_hit_offsets.layer_offset(planeCode);
    for ( int i_ohit = 0; i_ohit < n_hitCandidatesInLayers[planeCode]; ++i_ohit ) {
      const int ohit_index = hitCandidatesInLayers[planeCode][i_ohit];
      const float zohit  = ut_hits.zAtYEq0[layer_offset + ohit_index];
      
      if(zohit==zhit) continue;
      
      const float xohit = x_pos_layers[ planeCode ][ i_ohit];
      const float xextrap = xhit + txUT*(zhit-zohit);
      if( xohit-xextrap < -PrVeloUTConst::overlapTol) continue;
      if( xohit-xextrap > PrVeloUTConst::overlapTol) break;
      
      track.addLHCbID(ut_hits.LHCbID[layer_offset + ohit_index]);
      assert( track.hitsNum < VeloUTTracking::max_track_size);
      
      // -- only one overlap hit
      break;
    }
  }
  assert( n_tracks < VeloUTTracking::max_num_tracks );
  VeloUT_tracks[n_tracks] = track;
}

// ==============================================================================
// -- Finds the hits in a given layer within a certain range
// ==============================================================================
__host__ __device__ void findHits( 
  const uint lowerBoundSectorGroup,
  const uint upperBoundSectorGroup,
  UTHits& ut_hits,
  UTHitOffsets& ut_hit_offsets,
  uint layer_offset,
  const int i_layer,
  const float* ut_dxDy,
  const MiniState& myState, 
  const float xTolNormFact,
  const float invNormFact,
  int hitCandidatesInLayer[VeloUTTracking::max_hit_candidates_per_layer],
  int &n_hitCandidatesInLayer,
  float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer])
{
  const uint posBeg = ut_hit_offsets.sector_group_offset(lowerBoundSectorGroup) - layer_offset;
  const uint posEnd = ut_hit_offsets.sector_group_offset(upperBoundSectorGroup)
    + ut_hit_offsets.sector_group_number_of_hits(upperBoundSectorGroup) - layer_offset;

  const auto zInit = ut_hits.zAtYEq0[layer_offset + posBeg];
  const auto yApprox = myState.y + myState.ty * (zInit - myState.z);
  const auto xOnTrackProto = myState.x + myState.tx*(zInit - myState.z);
  const auto yyProto =       myState.y - myState.ty*myState.z;
  const float dxDy = ut_dxDy[i_layer];

  for (int i=posBeg; i<posEnd; ++i) {
    const auto xx = ut_hits.xAt(layer_offset + i, yApprox, dxDy); 
    const auto dx = xx - xOnTrackProto;
    
    if( dx < -xTolNormFact ) continue;
    if( dx >  xTolNormFact ) continue; 
    
    // -- Now refine the tolerance in Y
    if ( ut_hits.isNotYCompatible( layer_offset + i, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx*invNormFact)) ) continue;
    
    const auto zz = ut_hits.zAtYEq0[layer_offset + i]; 
    const auto yy = yyProto +  myState.ty*zz;
    const auto xx2 = ut_hits.xAt(layer_offset + i, yy, dxDy);
        
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


