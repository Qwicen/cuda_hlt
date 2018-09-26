#include "SearchWindow.cuh"
#include "BinarySearch.cuh"
#include "VeloTools.cuh"

//=============================================================================
// Reject tracks outside of acceptance or pointing to the beam pipe
//=============================================================================
__host__ __device__ bool velo_track_in_UTA_acceptance(
  const BasicState& state
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

// //=============================================================================
// // Search for the highest and lowest hit for the window
// //=============================================================================
// __device__ void binary_search_range(
//   const int layer,
//   const UTHits& ut_hits,
//   const UTHitOffsets& ut_hit_offsets,
//   const float ut_dxDy,
//   const float low_bound_x,
//   const float up_bound_x,
//   const float xTolNormFact,
//   const float yApprox,
//   const float xOnTrackProto,
//   const int layer_offset,
//   const uint first_sector_group_in_layer,
//   const uint last_sector_group_in_layer,
//   const float* dev_unique_sector_xs,
//   int& high_hit_pos,
//   int& low_hit_pos)
// {
//   // const int num_hits_layer = ut_hit_count.n_hits_layers[layer];
//   // const uint number_of_sector_groups = last_sector_group_in_layer - first_sector_group_in_layer;

//   uint min_x = first_sector_group_in_layer;
//   uint max_x = last_sector_group_in_layer - 1;
//   uint guess_x = 0;

//   uint guess_x = 0;

//   high_hit_pos = -1;
//   low_hit_pos = -1;

//   // // The window of search is out of bounds
//   // if (up_bound_x < dev_unique_sector_xs[first_sector_group_in_layer] ||
//   //     low_bound_x > dev_unique_sector_xs[last_sector_group_in_layer - 1]) {
//   //   // TODO return?
//   //   return;
//   // }

//   // look for the low limit
//   guess_x = (upperBoundSectorGroup + lowerBoundSectorGroup) / 2;
//   while (upperBoundSectorGroup != lowerBoundSectorGroup+1) {
//     const float xx = ut_hits.xAt(layer_offset + guess_x, yApprox, ut_dxDy);
//     const float dx = xx - xOnTrackProto;
//     if (dx < -xTolNormFact) {
//       lowerBoundSectorGroup = guess_x;
//     } else {
//       upperBoundSectorGroup = guess_x;
//     }   
//     guess_x = (upperBoundSectorGroup + lowerBoundSectorGroup) / 2;
//   }
//   low_hit_pos = guess_x;

//   // look for the high limit
//   lowerBoundSectorGroup = low_hit_pos;
//   upperBoundSectorGroup = last_sector_group_in_layer - 1; // last hit of the layer
//   guess_x = (upperBoundSectorGroup + lowerBoundSectorGroup) / 2;
//   while (upperBoundSectorGroup != lowerBoundSectorGroup+1) {
//     const float xx = ut_hits.xAt(layer_offset + guess_x, yApprox, ut_dxDy);
//     const float dx = xx - xOnTrackProto;
//     if (dx < xTolNormFact) {
//       upperBoundSectorGroup = guess_x;
//     } else {
//       lowerBoundSectorGroup = guess_x;
//     }   
//     guess_x = (upperBoundSectorGroup + lowerBoundSectorGroup) / 2;
//   }
//   high_hit_pos = guess_x;

//   assert(upperBoundSectorGroup < last_sector_group_in_layer);
//   assert(lowerBoundSectorGroup >= first_sector_group_in_layer);
//   assert(lowerBoundSectorGroup < upperBoundSectorGroup);

//   // Now binary search in Y
//   uint max_y;
//   uint min_y;
//   uint guess_y;

//   uint high_hit_pos_y = -1;
//   uint low_hit_pos_y = -1;

//   // look for the low limit
//   guess_y = (max_y + min_y) / 2;
//   while (max_y != min_y+1) {

//     const float xx = ut_hits.xAt(layer_offset + guess_y, yApprox, ut_dxDy);
//     const float dx = xx - xOnTrackProto;

//     const float tol = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx*invNormFact);
//     const float ymin = ut_hits.yMin(layer_offset + guess_y) - tol;
    
//     if (yApprox < ymin) {
//       min_y = guess_y;
//     } else {
//       max_y = guess_y;
//     }   
//     guess_y = (max_y + min_y) / 2;
//   }
//   low_hit_pos_y = guess_y;

//   // look for the high limit
//   min_y = low_hit_pos;
//   max_y = last_sector_group_in_layer - 1; // last hit of the layer
//   guess_y = (max_y + min_y) / 2;
//   while (max_y != min_y+1) {
//     const float xx = ut_hits.xAt(layer_offset + guess_y, yApprox, ut_dxDy);
//     const float dx = xx - xOnTrackProto;

//     const float tol = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx*invNormFact);
//     const float ymax = ut_hits.yMax(layer_offset + guess_y) + tol;

//     if (yApprox > ymax) {
//       max_y = guess_y;
//     } else {
//       min_y = guess_y;
//     }   
//     guess_y = (max_y + min_y) / 2;
//   }
//   high_hit_pos_y = guess_y;
// }

//=============================================================================
// Get the windows
//=============================================================================
__device__ void get_windows(
  const int i_track,
  const BasicState& veloState,
  const float* fudgeFactors,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_offsets,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets,
  const Velo::Consolidated::Tracks& velo_tracks,
  int* windows_layers) 
{
  // -- This is hardcoded, so faster
  // -- If you ever change the Table in the magnet tool, this will be wrong
  const float absSlopeY = std::abs( veloState.ty );
  const int index = (int)(absSlopeY*100 + 0.5);
  assert( 3 + 4*index < PrUTMagnetTool::N_dxLay_vals );
  const float normFact[4] {
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

    int layer_offset = ut_hit_offsets.layer_offset(layer);

    const float dx_dy   = ut_dxDy[layer];
    const float z_at_layer = ut_hits.zAtYEq0[layer_offset]; 
    const float y_at_z   = veloState.y + veloState.ty*(zLayer - veloState.z);
    const float x_at_layer = veloState.x + veloState.tx*(zLayer - veloState.z);
    const float y_at_layer = yAtZ + yTol * dxDyHelper[layer];

    const float zInit = ut_hits.zAtYEq0[layer_offset];
    const float y_track = veloState.y + veloState.ty * (zInit - veloState.z);
    const float x_track = veloState.x + veloState.tx * (zInit - veloState.z);
    const float invNormFact = 1.0/normFact[layer];

    // Find sector group for lowerBoundX and upperBoundX
    const uint first_sector_group_in_layer = dev_unique_x_sector_layer_offsets[layer];
    const uint last_sector_group_in_layer = dev_unique_x_sector_layer_offsets[layer+1];
    const uint sector_group_size = last_sector_group_in_layer - first_sector_group_in_layer;

    const uint sector_group = binary_search_leftmost(
      dev_unique_sector_xs + first_sector_group_in_layer,
      sector_group_size,
      x_at_layer);

    if (sector_group != 0) {
      const float x_at_left_sector = dev_unique_sector_xs[sector_group];
      const float x_at_right_sector = dev_unique_sector_xs[sector_group + 1];

      const float xx_at_left_sector = x_at_left_sector + y_track * ut_dxDy;
      const float xx_at_right_sector = x_at_right_sector + y_track * ut_dxDy;

      const float dx_max = std::max(xx_at_left_sector - x_track, xx_at_right_sector - x_track);

      const float tol = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx_max*invNormFact); 

      const uint sector_group_offset = ut_hit_offsets.sector_group_offset(sector_group);

      int first_candidate = -1, last_candidate = -1;
      first_candidate = binary_search_first_candidate(
        ut_hits.yEnd + sector_group_offset,
        ut_hit_offsets.sector_group_number_of_hits(sector_group),
        y_track,
        tol,
        [&ut_hits.yBegin, &sector_group_offset](
          const auto value, 
          const auto array_element, 
          const int index, 
          const float margin) {
            return (value + margin > ut_hits.yBegin[sector_group_offset + index] &&
              value - margin < array_element);
        }
      );
      if (first_candidate != -1) {
        last_candidate = binary_search_second_candidate(
          ut_hits.yBegin + sector_group_offset + first_candidate,
          ut_hit_offsets.sector_group_number_of_hits(sector_group) - first_candidate,
          y_track,
          tol
        );
        first_candidate += sector_group_offset;
        last_candidate = last_candidate==0 ? first_candidate+1 : first_candidate+last_candidate;
      }

      // TODO save first and last candidates in the correct position of windows_layers
      windows_layers[2 * velo_tracks.track_offset] = first_candidate;
      windows_layers[2 * velo_tracks.track_offset + 1] = last_candidate;
    }
  }
}