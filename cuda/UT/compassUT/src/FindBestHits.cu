#include "FindBestHits.cuh"

//=========================================================================
// Get the best 3 or 4 hits, 1 per layer, for a given VELO track
// When iterating over a panel, 3 windows are given, we set the index
// to be only in the windows
//=========================================================================
__device__ void find_best_hits(
  const int* win_size_shared,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const bool forward,
  int* best_hits,
  BestParams& best_params)
{
  // handle forward / backward cluster search
  int layers[N_LAYERS];
  #pragma unroll
  for (int i_layer = 0; i_layer < N_LAYERS; ++i_layer) {
    if (forward)
      layers[i_layer] = i_layer;
    else
      layers[i_layer] = N_LAYERS - 1 - i_layer;
  }

  const float yyProto = velo_state.y - velo_state.ty * velo_state.z;

  // // Get windows of all layers
  WindowIndicator win_ranges(win_size_shared); 
  const auto* ranges = win_ranges.get_track_candidates(threadIdx.x);

  int best_number_of_hits = 3;
  int temp_best_hits[N_LAYERS] = {-1, -1, -1, -1};
  bool found = false;
  int considered = 0;

  // loop over the 3 windows, putting the index in the windows
  // loop over layer 0
  float best_fit = PrVeloUTConst::maxPseudoChi2;
  for (int i0=0; (!found || considered < CompassUT::max_considered_before_found) &&
       i0<ranges->layer[layers[0]].size0 + ranges->layer[layers[0]].size1 + ranges->layer[layers[0]].size2; ++i0) {

    int i_hit0 = set_index(i0, ranges->layer[layers[0]]);

    // Get the hit to check with next layer
    const float yy0 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit0]);
    const float xhitLayer0 = ut_hits.xAt(i_hit0, yy0, ut_dxDy[layers[0]]);
    const float zhitLayer0 = ut_hits.zAtYEq0[i_hit0];
    temp_best_hits[0] = i_hit0;

    // loop over layer 2
    for (int i2=0; (!found || considered < CompassUT::max_considered_before_found) &&
         i2<ranges->layer[layers[2]].size0 + ranges->layer[layers[2]].size1 + ranges->layer[layers[2]].size2; ++i2) {

      int i_hit2 = set_index(i2, ranges->layer[layers[2]]);

      // Get the hit to check with next layer
      const float yy2 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit2]);
      const float xhitLayer2 = ut_hits.xAt(i_hit2, yy2, ut_dxDy[layers[2]]);
      const float zhitLayer2 = ut_hits.zAtYEq0[i_hit2];
      temp_best_hits[2] = i_hit2;

      const float tx = (xhitLayer2 - xhitLayer0) / (zhitLayer2 - zhitLayer0);
      if (std::abs(tx - velo_state.tx) > PrVeloUTConst::deltaTx2) continue;

      float hitTol = PrVeloUTConst::hitTol2;
      temp_best_hits[1] = -1;

      // search for triplet in layer1
      for (int i1=0; i1<ranges->layer[layers[1]].size0 + ranges->layer[layers[1]].size1 + ranges->layer[layers[1]].size2; ++i1) {

        int i_hit1 = set_index(i1, ranges->layer[layers[1]]);

        // Get the hit to check with next layer
        const float yy1 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit1]);
        const float xhitLayer1 = ut_hits.xAt(i_hit1, yy1, ut_dxDy[layers[1]]);
        const float zhitLayer1 = ut_hits.zAtYEq0[i_hit1];
        const float xextrapLayer1 = xhitLayer0 + tx * (zhitLayer1 - zhitLayer0);

        if (std::abs(xhitLayer1 - xextrapLayer1) < hitTol) {
          hitTol = std::abs(xhitLayer1 - xextrapLayer1);
          temp_best_hits[1] = i_hit1;
        }
      }

      // search for quadruplet in layer3
      temp_best_hits[3] = -1;
      hitTol = PrVeloUTConst::hitTol2;
      for (int i3=0; i3<ranges->layer[layers[3]].size0 + ranges->layer[layers[3]].size1 + ranges->layer[layers[3]].size2; ++i3) {

        int i_hit3 = set_index(i3, ranges->layer[layers[3]]);

        // Get the hit to check
        const float yy3 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit3]);
        const float xhitLayer3 = ut_hits.xAt(i_hit3, yy3, ut_dxDy[layers[3]]);
        const float zhitLayer3 = ut_hits.zAtYEq0[i_hit3];
        const float xextrapLayer3 = xhitLayer2 + tx * (zhitLayer3 - zhitLayer2);
        if (std::abs(xhitLayer3 - xextrapLayer3) < hitTol) {
          hitTol = std::abs(xhitLayer3 - xextrapLayer3);
          temp_best_hits[3] = i_hit3;
        }          
      }

      // Fit the hits to get q/p, chi2
      const auto temp_number_of_hits = 2 + (temp_best_hits[1] != -1) + (temp_best_hits[3] != -1);
      const auto params = pkick_fit(temp_best_hits, ut_hits, velo_state, ut_dxDy, yyProto, forward);
      ++considered;

      if (params.chi2UT < best_fit && temp_number_of_hits >= best_number_of_hits) {
        best_hits[0] = temp_best_hits[0];
        best_hits[1] = temp_best_hits[1];
        best_hits[2] = temp_best_hits[2];
        best_hits[3] = temp_best_hits[3];
        best_number_of_hits = temp_number_of_hits;
        best_params = params;
        best_fit = params.chi2UT;

        found = true;
      }
    }
  }
}

//=========================================================================
// apply the p-kick method to the triplet/quadruplet
// TODO return the chi2?
// TODO precalculate zDiff (its always the same)
//=========================================================================
__device__ BestParams pkick_fit(
  const int best_hits[N_LAYERS],
  const UTHits& ut_hits,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const float yyProto,
  const bool forward)
{
  BestParams best_params;

  // Helper stuff from velo state
  const float xMidField = velo_state.x + velo_state.tx * (PrVeloUTConst::zKink - velo_state.z);
  const float a = PrVeloUTConst::sigmaVeloSlope * (PrVeloUTConst::zKink - velo_state.z);
  const float wb = 1.0f / (a * a);

  float mat[3] = {wb, wb * PrVeloUTConst::zDiff, wb * PrVeloUTConst::zDiff * PrVeloUTConst::zDiff};
  float rhs[2] = {wb * xMidField, wb * xMidField * PrVeloUTConst::zDiff};

  // add hits
  #pragma unroll
  for (int i = 0; i < N_LAYERS; ++i) {
    int hit_index = best_hits[i];
    if (hit_index >= 0) {
      const float wi = ut_hits.weight[hit_index];
      const int plane_code = forward ? i : N_LAYERS - 1 - i;
      const float dxDy = ut_dxDy[plane_code];
      const float ci = ut_hits.cosT(hit_index, dxDy);
      const float dz = 0.001f * (ut_hits.zAtYEq0[hit_index] - PrVeloUTConst::zMidUT);
      // x_pos_layer
      const float yy = yyProto + (velo_state.ty * ut_hits.zAtYEq0[hit_index]);
      const float ui = ut_hits.xAt(hit_index, yy, dxDy);

      mat[0] += wi * ci;
      mat[1] += wi * ci * dz;
      mat[2] += wi * ci * dz * dz;
      rhs[0] += wi * ui;
      rhs[1] += wi * ui * dz;
    }
  }

  const float denom = 1.0f / (mat[0] * mat[2] - mat[1] * mat[1]);
  const float xSlopeUTFit = 0.001f * (mat[0] * rhs[1] - mat[1] * rhs[0]) * denom;
  const float xUTFit = (mat[2] * rhs[0] - mat[1] * rhs[1]) * denom;

  // new VELO slope x
  const float xb = xUTFit + xSlopeUTFit * (PrVeloUTConst::zKink - PrVeloUTConst::zMidUT);
  const float invKinkVeloDist = 1 / (PrVeloUTConst::zKink - velo_state.z);
  const float xSlopeVeloFit = (xb - velo_state.x) * invKinkVeloDist;
  const float chi2VeloSlope = (velo_state.tx - xSlopeVeloFit) * PrVeloUTConst::invSigmaVeloSlope;

  // chi2 takes chi2 from velo fit + chi2 from UT fit
  float chi2UT = chi2VeloSlope * chi2VeloSlope;
  // add chi2
  int total_num_hits = 0;
  #pragma unroll
  for (int i = 0; i < N_LAYERS; ++i) {
    int hit_index = best_hits[i];
    if (hit_index >= 0) {
      const float zd = ut_hits.zAtYEq0[hit_index];
      const float xd = xUTFit + xSlopeUTFit * (zd - PrVeloUTConst::zMidUT);
      // x_pos_layer
      const int plane_code = forward ? i : N_LAYERS - 1 - i;
      const float dxDy = ut_dxDy[plane_code];
      const float yy = yyProto + (velo_state.ty * ut_hits.zAtYEq0[hit_index]);
      const float x = ut_hits.xAt(hit_index, yy, dxDy);

      const float du = xd - x;
      chi2UT += (du * du) * ut_hits.weight[hit_index];

      // count the number of processed htis
      total_num_hits++;
    }
  }

  chi2UT /= (total_num_hits - 1);

  // Save the best parameters if chi2 is good
  if (chi2UT < PrVeloUTConst::maxPseudoChi2) {
    // calculate q/p
    const float sinInX = xSlopeVeloFit * std::sqrt(1.0f + xSlopeVeloFit * xSlopeVeloFit);
    const float sinOutX = xSlopeUTFit * std::sqrt(1.0f + xSlopeUTFit * xSlopeUTFit);

    best_params.qp = sinInX - sinOutX;
    best_params.chi2UT = chi2UT;
    best_params.n_hits = total_num_hits;
  }

  return best_params;
}

__device__ __inline__ int set_index(
  const int i, 
  const LayerCandidates& layer_cand)
{
  int hit = 0;
  if (i < layer_cand.size0) {
    hit = layer_cand.from0 + i;
  } else if (i < layer_cand.size0 + layer_cand.size1) {
    hit = layer_cand.from1 + i - layer_cand.size0;
  } else {
    hit = layer_cand.from2 + i - layer_cand.size0- layer_cand.size1;
  }
  return hit;
}

//=========================================================================
// Check if hit is inside tolerance and refine by Y
//=========================================================================
__host__ __device__ __inline__ bool check_tol_refine(
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