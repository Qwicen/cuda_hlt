#include "BinarySearch.cuh"
#include "VeloTools.cuh"
#include "CalculateWindows.cuh"
#include "BinarySearchFirstCandidate.cuh"

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

//=========================================================================
// Check if hit is inside tolerance and refine by Y
//=========================================================================
__host__ __device__ __inline__ bool is_valid_tol_refine (
  const int hit_index,
  const UTHits& ut_hits,
  const MiniState& velo_state,
  const float normFactNum,
  const float xTol,
  const float dxDy)
{
  const float xTolNormFact = xTol * (1.0f / normFactNum);

  const float zInit = ut_hits.zAtYEq0[hit_index];
  const float yApprox = velo_state.y + velo_state.ty * (zInit - velo_state.z);
  const float xOnTrackProto = velo_state.x + velo_state.tx * (zInit - velo_state.z);

  const float xx = ut_hits.xAt(hit_index, yApprox, dxDy);
  const float dx = xx - xOnTrackProto;

  if (dx < -xTolNormFact || dx > xTolNormFact) return false;

  // Now refine the tolerance in Y
  if (ut_hits.isNotYCompatible(
        hit_index, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx * (1.0f / normFactNum))))
    return false;

  return true;
}

//=============================================================================
// Get the windows
//=============================================================================
__device__ std::tuple<int, int, int, int, int, int> calculate_windows(
  const int i_track,
  const int layer,
  const MiniState& velo_state,
  const float* fudge_factors,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_offsets,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets,
  const Velo::Consolidated::Tracks& velo_tracks)
{
  // -- This is hardcoded, so faster
  // -- If you ever change the Table in the magnet tool, this will be wrong
  const float absSlopeY = std::abs(velo_state.ty);
  const int index       = (int) (absSlopeY * 100 + 0.5f);
  assert(3 + 4 * index < PrUTMagnetTool::N_dxLay_vals);
  const float normFact[4]{
    fudge_factors[4 * index], fudge_factors[1 + 4 * index], fudge_factors[2 + 4 * index], fudge_factors[3 + 4 * index]};

  // -- this 500 seems a little odd...
  // to do: change back!
  const float invTheta = std::min(500.0f, 1.0f / std::sqrt(velo_state.tx * velo_state.tx + velo_state.ty * velo_state.ty));
  const float minMom   = std::max(PrVeloUTConst::minPT * invTheta, 1.5f * Gaudi::Units::GeV);
  const float xTol     = std::abs(1.0f / (PrVeloUTConst::distToMomentum * minMom));
  // const float yTol     = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * xTol;

  int layer_offset = ut_hit_offsets.layer_offset(layer);

  const float dx_dy      = ut_dxDy[layer];
  const float z_at_layer = ut_hits.zAtYEq0[layer_offset];
  const float y_track     = velo_state.y + velo_state.ty * (z_at_layer - velo_state.z);
  const float x_track     = velo_state.x + velo_state.tx * (z_at_layer - velo_state.z);
  const float invNormFact = 1.0f / normFact[layer];

  // Second sector group search
  // const float tolerance_in_x = xTol * invNormFact;

  // Find sector group for lowerBoundX and upperBoundX
  const int first_sector_group_in_layer = dev_unique_x_sector_layer_offsets[layer];
  const int last_sector_group_in_layer  = dev_unique_x_sector_layer_offsets[layer + 1];
  const int sector_group_size           = last_sector_group_in_layer - first_sector_group_in_layer;

  const int local_sector_group =
    binary_search_leftmost(dev_unique_sector_xs + first_sector_group_in_layer, sector_group_size, x_track);
  int sector_group = first_sector_group_in_layer + local_sector_group;

  int first_candidate = -1, last_candidate = -1;
  int left_group_first_candidate = -1, left_group_last_candidate = -1;
  int right_group_first_candidate = -1, right_group_last_candidate = -1;
  if (sector_group != 0) {
    // The sector we are interested on is sector_group - 1
    sector_group -= 1;
    const auto sector_candidates = find_candidates_in_sector_group(
      ut_hits,
      ut_hit_offsets,
      velo_state,
      dev_unique_sector_xs,
      x_track,
      y_track,
      dx_dy,
      normFact[layer],
      invNormFact,
      xTol,
      sector_group
    );

    first_candidate = std::get<0>(sector_candidates);
    last_candidate = std::get<1>(sector_candidates);

    // Left group
    const int left_group = sector_group - 1;
    if (left_group >= first_sector_group_in_layer) {
      // We found a sector group with potentially compatible hits
      // Look for them
      const auto left_group_candidates = find_candidates_in_sector_group(
        ut_hits,
        ut_hit_offsets,
        velo_state,
        dev_unique_sector_xs,
        x_track,
        y_track,
        dx_dy,
        normFact[layer],
        invNormFact,
        xTol,
        left_group
      );

      left_group_first_candidate = std::get<0>(left_group_candidates);
      left_group_last_candidate = std::get<1>(left_group_candidates);
    }

    // Right group
    const int right_group = sector_group + 1;
    if (right_group < last_sector_group_in_layer) {
      // We found a sector group with potentially compatible hits
      // Look for them
      const auto right_group_candidates = find_candidates_in_sector_group(
        ut_hits,
        ut_hit_offsets,
        velo_state,
        dev_unique_sector_xs,
        x_track,
        y_track,
        dx_dy,
        normFact[layer],
        invNormFact,
        xTol,
        right_group
      );

      right_group_first_candidate = std::get<0>(right_group_candidates);
      right_group_last_candidate = std::get<1>(right_group_candidates);
    }
  }

  return {first_candidate, last_candidate,
    left_group_first_candidate, left_group_last_candidate,
    right_group_first_candidate, right_group_last_candidate
  };
} 

__device__ std::tuple<int, int> find_candidates_in_sector_group(
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_offsets,
  const MiniState& velo_state,
  const float* dev_unique_sector_xs,
  const float x_track,
  const float y_track,
  const float dx_dy,
  const float normFact,
  const float invNormFact,
  const float xTol,
  const int sector_group)
{
  const float x_at_left_sector  = dev_unique_sector_xs[sector_group];
  const float x_at_right_sector = dev_unique_sector_xs[sector_group + 1];
  const float xx_at_left_sector  = x_at_left_sector + y_track * dx_dy;
  const float xx_at_right_sector = x_at_right_sector + y_track * dx_dy;
  const float dx_max = std::max(xx_at_left_sector - x_track, xx_at_right_sector - x_track);

  const float tol = PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx_max * invNormFact);
  const uint sector_group_offset = ut_hit_offsets.sector_group_offset(sector_group);

  int first_candidate = -1, last_candidate = -1;
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

    bool found = false;

    // refine first candidate
    for (int i=first_candidate; i<last_candidate; ++i) {
      if (is_valid_tol_refine(i, ut_hits, velo_state, normFact, xTol, dx_dy)) {
        first_candidate = i;
        found = true;
        break;
      }
    }

    if (!found) {
      first_candidate = -1;
      last_candidate = -1;      
    } 
    else {
      bool last_found = false;

      for (int i=last_candidate; i>first_candidate; --i) {
        if (is_valid_tol_refine(i, ut_hits, velo_state, normFact, xTol,dx_dy)) {
          last_candidate = i;
          last_found = true;
          break;
        } 
      }

      if (!last_found) {
        last_candidate = first_candidate + 1;
      }      
    }
  }

  return {first_candidate, last_candidate};
}
