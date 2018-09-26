#include "BinarySearch.cuh"
#include "VeloTools.cuh"
#include "CalculateWindows.cuh"
#include "BinarySearchFirstCandidate.cuh"
#include <tuple>

//=============================================================================
// Reject tracks outside of acceptance or pointing to the beam pipe
//=============================================================================
__device__ bool velo_track_in_UTA_acceptance(const MiniState& state)
{
  const float xMidUT = state.x + state.tx * (PrVeloUTConst::zMidUT - state.z);
  const float yMidUT = state.y + state.ty * (PrVeloUTConst::zMidUT - state.z);

  if (xMidUT * xMidUT + yMidUT * yMidUT < PrVeloUTConst::centralHoleSize * PrVeloUTConst::centralHoleSize) return false;
  if ((std::abs(state.tx) > PrVeloUTConst::maxXSlope) || (std::abs(state.ty) > PrVeloUTConst::maxYSlope)) return false;

  if (
    PrVeloUTConst::passTracks && std::abs(xMidUT) < PrVeloUTConst::passHoleSize &&
    std::abs(yMidUT) < PrVeloUTConst::passHoleSize) {
    return false;
  }

  return true;
}

//=============================================================================
// Get the windows
//=============================================================================
__device__ std::tuple<int, int> calculate_windows(
  const int i_track,
  const int layer,
  const MiniState& veloState,
  const float* fudgeFactors,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_offsets,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets,
  const Velo::Consolidated::Tracks& velo_tracks)
{
  // -- This is hardcoded, so faster
  // -- If you ever change the Table in the magnet tool, this will be wrong
  const float absSlopeY = std::abs(veloState.ty);
  const int index       = (int) (absSlopeY * 100 + 0.5);
  assert(3 + 4 * index < PrUTMagnetTool::N_dxLay_vals);
  const float normFact[4]{
    fudgeFactors[4 * index], fudgeFactors[1 + 4 * index], fudgeFactors[2 + 4 * index], fudgeFactors[3 + 4 * index]};

  // -- this 500 seems a little odd...
  // to do: change back!
  const float invTheta = std::min(500., 1.0 / std::sqrt(veloState.tx * veloState.tx + veloState.ty * veloState.ty));
  const float minMom   = std::max(PrVeloUTConst::minPT * invTheta, float(1.5) * Gaudi::Units::GeV);
  const float xTol     = std::abs(1. / (PrVeloUTConst::distToMomentum * minMom));
  const float yTol     = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * xTol;

  int layer_offset = ut_hit_offsets.layer_offset(layer);

  const float dx_dy      = ut_dxDy[layer];
  const float z_at_layer = ut_hits.zAtYEq0[layer_offset];
  const float y_track     = veloState.y + veloState.ty * (z_at_layer - veloState.z);
  const float x_track     = veloState.x + veloState.tx * (z_at_layer - veloState.z);
  const float invNormFact = 1.0 / normFact[layer];

  // Find sector group for lowerBoundX and upperBoundX
  const uint first_sector_group_in_layer = dev_unique_x_sector_layer_offsets[layer];
  const uint last_sector_group_in_layer  = dev_unique_x_sector_layer_offsets[layer + 1];
  const uint sector_group_size           = last_sector_group_in_layer - first_sector_group_in_layer;

  const uint sector_group =
    binary_search_leftmost(dev_unique_sector_xs + first_sector_group_in_layer, sector_group_size, x_track);

  int first_candidate = -1, last_candidate = -1;
  if (sector_group != 0) {
    const float x_at_left_sector  = dev_unique_sector_xs[sector_group];
    const float x_at_right_sector = dev_unique_sector_xs[sector_group + 1];

    const float xx_at_left_sector  = x_at_left_sector + y_track * dx_dy;
    const float xx_at_right_sector = x_at_right_sector + y_track * dx_dy;

    const float dx_max = std::max(xx_at_left_sector - x_track, xx_at_right_sector - x_track);

    const float tol = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx_max * invNormFact);

    const uint sector_group_offset = ut_hit_offsets.sector_group_offset(sector_group);

    first_candidate = binary_search_first_candidate(
      ut_hits.yEnd + sector_group_offset,
      ut_hit_offsets.sector_group_number_of_hits(sector_group),
      y_track,
      tol,
      [&] (const auto value, const auto array_element, const int index, const float margin) {
        return (value + margin > ut_hits.yBegin[sector_group_offset + index] && value - margin < array_element);
      });
    if (first_candidate != -1) {
      last_candidate = binary_search_second_candidate(
        ut_hits.yBegin + sector_group_offset + first_candidate,
        ut_hit_offsets.sector_group_number_of_hits(sector_group) - first_candidate,
        y_track,
        tol);
      first_candidate += sector_group_offset;
      last_candidate = last_candidate == 0 ? first_candidate + 1 : first_candidate + last_candidate;
    }
  }

  return {first_candidate, last_candidate};
}