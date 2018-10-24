#include "CompassUT.cuh"

#include "CalculateWindows.cuh"
#include "BinarySearchFirstCandidate.cuh"
#include "BinarySearch.cuh"

__global__ void compass_ut(
  uint* dev_ut_hits, // actual hit content
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage, // semi_prefixsum, offset to tracks
  uint* dev_velo_track_hit_number,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  int* dev_active_tracks,
  const uint* dev_unique_x_sector_layer_offsets, // prefixsum to point to the x hit of the sector, per layer
  const float* dev_unique_sector_xs, // list of xs that define the groups
  VeloUTTracking::TrackUT* dev_compassUT_tracks,
  int* dev_atomics_compassUT, // size of number of events
  int* dev_windows_layers)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[N_LAYERS];
  const uint total_number_of_hits = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks{
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states{dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  UTHitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  UTHits ut_hits {dev_ut_hits, total_number_of_hits};

  // active track pointer
  // int* active_tracks = dev_active_tracks + event_number;

  // dev_atomics_compassUT contains in an SoA:
  //   1. # of veloUT tracks
  //   2. # velo tracks in UT acceptance
  // This is to write the final track
  int* n_veloUT_tracks_event = dev_atomics_compassUT + event_number;
  VeloUTTracking::TrackUT* veloUT_tracks_event = dev_compassUT_tracks + event_number * VeloUTTracking::max_num_tracks;

  // initialize atomic veloUT tracks counter && active track
  if (threadIdx.x == 0) {
    *n_veloUT_tracks_event = 0;
    // *active_tracks         = 0;
  }

  // int shared_active_tracks[2 * VeloUTTracking::num_threads - 1];

  // __syncthreads();

  const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  const float* bdl_table = &(dev_ut_magnet_tool->bdlTable[0]);

  for (int i_track = threadIdx.x; i_track < number_of_tracks_event; i_track += blockDim.x) {

    const uint current_track_offset = event_tracks_offset + i_track;

    // TODO the non active tracks should be -1
    // const int i_track = shared_active_tracks[threadIdx.x];

    // select velo track to join with UT hits
    const uint velo_states_index = event_tracks_offset + i_track;
    const MiniState velo_state{velo_states, velo_states_index};

    // if (i_track >= number_of_tracks_event) continue;
    // if (velo_states.backward[velo_states_index]) continue;
    // if(!velo_track_in_UT_acceptance(velo_state)) continue;    

    int best_hits[N_LAYERS] = {-1, -1, -1, -1};
    BestParams best_params;

    // Find compatible hits in the windows for this VELO track
    find_best_hits(
      i_track,
      current_track_offset,
      dev_windows_layers,
      ut_hits,
      ut_hit_offsets,
      velo_state,
      fudgeFactors,
      dev_ut_dxDy,
      true,
      best_hits,
      best_params);

    // Count found hits
    int total_num_hits = 0;
    #pragma unroll
    for (int i = 0; i < N_LAYERS; ++i) {
      if (best_hits[i] >= 0) total_num_hits++;
    }

    // write the final track
    if (total_num_hits >= (N_LAYERS - 1)) {
      save_track(
        i_track,
        bdl_table,
        velo_state,
        best_params,
        dev_velo_track_hits,
        velo_tracks,
        total_num_hits,
        best_hits,
        ut_hits,
        dev_ut_dxDy,
        n_veloUT_tracks_event,
        veloUT_tracks_event);
    }

    //     const int j = blockDim.x + threadIdx.x;
    //     if (j < *active_tracks) {
    //       shared_active_tracks[threadIdx.x] = shared_active_tracks[j];
    //     }

    //     __syncthreads();

    //     if (threadIdx.x == 0) {
    //       *active_tracks -= blockDim.x;
    //     }
  }

  // // remaining tracks
  // if (threadIdx.x < *active_tracks) {

  //   // store a window(2 positions) for each layer, for each thrack
  //   __shared__ int windows_layers[VeloUTTracking::num_threads * VeloUTTracking::n_layers * 2];

  //   const int i_track = shared_active_tracks[threadIdx.x];

  // }
}

//=============================================================================
// Reject tracks outside of acceptance or pointing to the beam pipe
//=============================================================================
__host__ __device__ bool velo_track_in_UT_acceptance(const MiniState& state)
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
__host__ __device__ __inline__ bool check_tol_refine(
  const int hit_index,
  const UTHits& ut_hits,
  const MiniState& velo_state,
  const float normFactNum,
  const float xTol,
  const float dxDy)
{
  bool valid_hit = true;

  const float xTolNormFact = xTol * (1.0f / normFactNum);

  const float zInit = ut_hits.zAtYEq0[hit_index];
  const float yApprox = velo_state.y + velo_state.ty * (zInit - velo_state.z);
  const float xOnTrackProto = velo_state.x + velo_state.tx * (zInit - velo_state.z);

  const float xx = ut_hits.xAt(hit_index, yApprox, dxDy);
  const float dx = xx - xOnTrackProto;

  if (dx < -xTolNormFact || dx > xTolNormFact) valid_hit = false;

  // Now refine the tolerance in Y
  if (ut_hits.isNotYCompatible(
        hit_index, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx * (1.0f / normFactNum))))
    valid_hit = false;

  return valid_hit;
}

__host__ __device__ __inline__ int set_index(
  const int i, const int from0, const int from1, const int from2, const int num_cand_0, const int num_cand_1)
{
  int hit = 0;
  if (i < num_cand_0) {
    hit = from0 + i;
  } else if (i < num_cand_0 + num_cand_1) {
    hit = from1 + i - num_cand_0;
  } else {
    hit = from2 + i - num_cand_0 - num_cand_1;
  }
  return hit;
}

//=========================================================================
// Get the best 3 or 4 hits, 1 per layer, for a given VELO track
// When iterating over a panel, 3 windows are given, we set the index
// to be only in the windows
//=========================================================================
__host__ __device__ void find_best_hits(
  const int i_track,
  const uint current_track_offset,
  const int* dev_windows_layers,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const MiniState& velo_state,
  const float* fudgeFactors,
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

  const float invTheta = std::min(500.0f, 1.0f / std::sqrt(velo_state.tx * velo_state.tx + velo_state.ty * velo_state.ty));
  const float minMom = std::max(PrVeloUTConst::minPT * invTheta, 1.5f * Gaudi::Units::GeV);
  const float xTol = std::abs(1. / ( PrVeloUTConst::distToMomentum * minMom ));
  const float yyProto = velo_state.y - velo_state.ty * velo_state.z;

  const float absSlopeY = std::abs( velo_state.ty );
  const int index = (int)(absSlopeY*100 + 0.5f);
  assert( 3 + 4*index < PrUTMagnetTool::N_dxLay_vals );  
  const std::array<float,4> normFact = { 
    fudgeFactors[4*index], 
    fudgeFactors[1 + 4*index], 
    fudgeFactors[2 + 4*index], 
    fudgeFactors[3 + 4*index] 
  };  

  // // Get windows of all layers
  // WindowIndicator win_ranges(dev_windows_layers); 
  // const auto* ranges = win_ranges.get_track_candidates(i_track);

  const int from_l0_g0 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 0];
  const int to_l0_g0 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 0 + 1];
  const int from_l0_g1 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 0 + 2];
  const int to_l0_g1 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 0 + 3];
  const int from_l0_g2 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 0 + 4];
  const int to_l0_g2 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 0 + 5];

  const int from_l1_g0 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 1];
  const int to_l1_g0 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 1 + 1];
  const int from_l1_g1 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 1 + 2];
  const int to_l1_g1 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 1 + 3];
  const int from_l1_g2 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 1 + 4];
  const int to_l1_g2 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 1 + 5];

  const int from_l2_g0 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 2];
  const int to_l2_g0 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 2 + 1];
  const int from_l2_g1 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 2 + 2];
  const int to_l2_g1 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 2 + 3];
  const int from_l2_g2 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 2 + 4];
  const int to_l2_g2 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 2 + 5];

  const int from_l3_g0 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 3];
  const int to_l3_g0 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 3 + 1];
  const int from_l3_g1 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 3 + 2];
  const int to_l3_g1 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 3 + 3];
  const int from_l3_g2 = dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 3 + 4];
  const int to_l3_g2 =   dev_windows_layers[6 * N_LAYERS * current_track_offset + 6 * 3 + 5];

  // Check number of hits in the windows
  const int num_candidates_l0_g0 = to_l0_g0 - from_l0_g0;
  const int num_candidates_l0_g1 = to_l0_g1 - from_l0_g1;
  const int num_candidates_l0_g2 = to_l0_g2 - from_l0_g2;

  const int num_candidates_l1_g0 = to_l1_g0 - from_l1_g0;
  const int num_candidates_l1_g1 = to_l1_g1 - from_l1_g1;
  const int num_candidates_l1_g2 = to_l1_g2 - from_l1_g2;

  const int num_candidates_l2_g0 = to_l2_g0 - from_l2_g0;
  const int num_candidates_l2_g1 = to_l2_g1 - from_l2_g1;
  const int num_candidates_l2_g2 = to_l2_g2 - from_l2_g2;

  const int num_candidates_l3_g0 = to_l3_g0 - from_l3_g0;
  const int num_candidates_l3_g1 = to_l3_g1 - from_l3_g1;
  const int num_candidates_l3_g2 = to_l3_g2 - from_l3_g2;

  // bool fourLayerSolution = false;

  // loop over the 3 windows, putting the index in the windows
  // loop over layer 0
  for (int i0=0; i0<num_candidates_l0_g0 + num_candidates_l0_g1 + num_candidates_l0_g2; ++i0) {

    int i_hit0 = set_index(i0, from_l0_g0, from_l0_g1, from_l0_g2, num_candidates_l0_g0, num_candidates_l0_g1);

    if (!check_tol_refine(
      i_hit0,
      ut_hits,
      velo_state,
      normFact[ut_hits.planeCode[i_hit0]],
      xTol,
      ut_dxDy[ut_hits.planeCode[i_hit0]])
    ) continue;

    // Get the hit to check with next layer
    const float yy0 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit0]);
    const float xhitLayer0 = ut_hits.xAt(i_hit0, yy0, ut_dxDy[layers[0]]);
    const float zhitLayer0 = ut_hits.zAtYEq0[i_hit0];
    best_hits[0] = i_hit0;

    // loop over layer 2
    for (int i2=0; i2<num_candidates_l2_g0 + num_candidates_l2_g1 + num_candidates_l2_g2; ++i2) {

      int i_hit2 = set_index(i2, from_l2_g0, from_l2_g1, from_l2_g2, num_candidates_l2_g0, num_candidates_l2_g1);

      if (!check_tol_refine(
        i_hit2,
        ut_hits,
        velo_state,
        normFact[ut_hits.planeCode[i_hit2]],
        xTol,
        ut_dxDy[ut_hits.planeCode[i_hit2]])
      ) continue;

      // Get the hit to check with next layer
      const float yy2 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit2]);
      const float xhitLayer2 = ut_hits.xAt(i_hit2, yy2, ut_dxDy[layers[2]]);
      const float zhitLayer2 = ut_hits.zAtYEq0[i_hit2];
      best_hits[2] = i_hit2;

      const float tx = (xhitLayer2 - xhitLayer0) / (zhitLayer2 - zhitLayer0);
      if (std::abs(tx - velo_state.tx) > PrVeloUTConst::deltaTx2) continue;

      float hitTol = PrVeloUTConst::hitTol2;

      // search for triplet in layer1
      for (int i1=0; i1<num_candidates_l1_g0 + num_candidates_l1_g1 + num_candidates_l1_g2; ++i1) {

        int i_hit1 = set_index(i1, from_l1_g0, from_l1_g1, from_l1_g2, num_candidates_l1_g0, num_candidates_l1_g1);

        if (!check_tol_refine(
          i_hit1,
          ut_hits,
          velo_state,
          normFact[ut_hits.planeCode[i_hit1]],
          xTol,
          ut_dxDy[ut_hits.planeCode[i_hit1]])
        ) continue;

        // Get the hit to check with next layer
        const float yy1 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit1]);
        const float xhitLayer1 = ut_hits.xAt(i_hit1, yy1, ut_dxDy[layers[1]]);
        const float zhitLayer1 = ut_hits.zAtYEq0[i_hit1];
        const float xextrapLayer1 = xhitLayer0 + tx * (zhitLayer1 - zhitLayer0);

        if (std::abs(xhitLayer1 - xextrapLayer1) < hitTol) {
          hitTol = std::abs(xhitLayer1 - xextrapLayer1);
          best_hits[1] = i_hit1;
        }
      }

      // search for quadruplet in layer3
      hitTol = PrVeloUTConst::hitTol2;
      for (int i3=0; i3<num_candidates_l3_g0 + num_candidates_l3_g1 + num_candidates_l3_g2; ++i3) {

        int i_hit3 = set_index(i3, from_l3_g0, from_l3_g1, from_l3_g2, num_candidates_l3_g0, num_candidates_l3_g1);

        if (!check_tol_refine(
          i_hit3,
          ut_hits,
          velo_state,
          normFact[ut_hits.planeCode[i_hit3]],
          xTol,
          ut_dxDy[ut_hits.planeCode[i_hit3]])
        ) continue;

        // Get the hit to check
        const float yy3 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit3]);
        const float xhitLayer3 = ut_hits.xAt(i_hit3, yy3, ut_dxDy[layers[3]]);
        const float zhitLayer3 = ut_hits.zAtYEq0[i_hit3];
        const float xextrapLayer3 = xhitLayer2 + tx * (zhitLayer3 - zhitLayer2);
        if (std::abs(xhitLayer3 - xextrapLayer3) < hitTol) {
          hitTol = std::abs(xhitLayer3 - xextrapLayer3);
          best_hits[3] = i_hit3;
        }          
      }

      // Fit the hits to get q/p, chi2
      best_params = pkick_fit(best_hits, ut_hits, velo_state, ut_dxDy, yyProto);

      // int num_hits = 0;
      // for (int i=0; i<N_LAYERS; ++i) { 
      //   if (best_hits[i] != -1) num_hits++; 
      // }

      // if(!fourLayerSolution && num_hits > 0) fourLayerSolution = true;
    }
  }
}

//=========================================================================
// apply the p-kick method to the triplet/quadruplet
// TODO return the chi2?
// TODO precalculate zDiff (its always the same)
//=========================================================================
__host__ __device__ BestParams pkick_fit(
  const int best_hits[N_LAYERS],
  const UTHits& ut_hits,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const float yyProto)
{
  BestParams best_params;

  // accumulate the high threshold
  int n_high_thres = 0;
  #pragma unroll
  for (int i = 0; i < N_LAYERS; ++i) {
    if (best_hits[i] >= 0) { n_high_thres += ut_hits.highThreshold[best_hits[i]]; }
  }

  // Veto hit combinations with no high threshold hit
  // = likely spillover
  if (n_high_thres < PrVeloUTConst::minHighThres) return best_params;

  // Scale the z-component, to not run into numerical problems with floats
  // first add to sum values from hit at xMidField, zMidField hit
  const float zDiff = 0.001f * (PrVeloUTConst::zKink - PrVeloUTConst::zMidUT);

  // Helper stuff from velo state
  const float xMidField = velo_state.x + velo_state.tx * (PrVeloUTConst::zKink - velo_state.z);
  const float a = PrVeloUTConst::sigmaVeloSlope * (PrVeloUTConst::zKink - velo_state.z);
  const float wb = 1.0f / (a * a);

  float mat[3] = {wb, wb * zDiff, wb * zDiff * zDiff};
  float rhs[2] = {wb * xMidField, wb * xMidField * zDiff};

  // add hits
  #pragma unroll
  for (int i = 0; i < N_LAYERS; ++i) {
    int hit_index = best_hits[i];
    if (hit_index >= 0) {

      const float wi = ut_hits.weight[hit_index];
      const float dxDy = ut_dxDy[ut_hits.planeCode[hit_index]];
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
      const float dxDy = ut_dxDy[ut_hits.planeCode[hit_index]];
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
    best_params.xUTFit = xUTFit;
    best_params.xSlopeUTFit = xSlopeUTFit;
  }

  return best_params;
}

// These things are all hardcopied from the PrTableForFunction and PrUTMagnetTool
// If the granularity or whatever changes, this will give wrong results
__host__ __device__ int master_index(const int index1, const int index2, const int index3)
{
  return (index3 * 11 + index2) * 31 + index1;
}

//=========================================================================
// prepare the final track
//=========================================================================
__device__ void save_track(
  const int i_track,
  const float* bdl_table,
  const MiniState& velo_state,
  const BestParams& best_params,
  uint* dev_velo_track_hits,
  const Velo::Consolidated::Tracks& velo_tracks,
  const int num_best_hits,
  const int* best_hits,
  const UTHits& ut_hits,
  const float* ut_dxDy,
  int* n_veloUT_tracks, // increment number of tracks
  VeloUTTracking::TrackUT VeloUT_tracks[VeloUTTracking::max_num_tracks]) // write the track
{
  //== Handle states. copy Velo one, add UT.
  const float zOrigin = (std::fabs(velo_state.ty) > 0.001f) ? velo_state.z - velo_state.y / velo_state.ty
                                                           : velo_state.z - velo_state.x / velo_state.tx;

  // -- These are calculations, copied and simplified from PrTableForFunction
  const float var[3] = {velo_state.ty, zOrigin, velo_state.z};

  const int index1 = std::max(0, std::min(30, int((var[0] + 0.3f) / 0.6f * 30)));
  const int index2 = std::max(0, std::min(10, int((var[1] + 250) / 500 * 10)));
  const int index3 = std::max(0, std::min(10, int(var[2] / 800 * 10)));

  assert(master_index(index1, index2, index3) < PrUTMagnetTool::N_bdl_vals);
  float bdl = bdl_table[master_index(index1, index2, index3)];

  const int num_idx = 3;
  const float bdls[num_idx] = {bdl_table[master_index(index1 + 1, index2, index3)],
                               bdl_table[master_index(index1, index2 + 1, index3)],
                               bdl_table[master_index(index1, index2, index3 + 1)]};
  const float deltaBdl[num_idx] = {0.02f, 50.0f, 80.0f};
  const float boundaries[num_idx] = {
    -0.3f + float(index1) * deltaBdl[0], -250.0f + float(index2) * deltaBdl[1], 0.0f + float(index3) * deltaBdl[2]};

  // This is an interpolation, to get a bit more precision
  float addBdlVal = 0.0f;
  const float minValsBdl[num_idx] = {-0.3f, -250.0f, 0.0f};
  const float maxValsBdl[num_idx] = {0.3f, 250.0f, 800.0f};
  for (int i = 0; i < num_idx; ++i) {
    if (var[i] < minValsBdl[i] || var[i] > maxValsBdl[i]) continue;
    const float dTab_dVar = (bdls[i] - bdl) / deltaBdl[i];
    const float dVar = (var[i] - boundaries[i]);
    addBdlVal += dTab_dVar * dVar;
  }
  bdl += addBdlVal;

  const float qpxz2p = -1 * std::sqrt(1.0f + velo_state.ty * velo_state.ty) / bdl * 3.3356f / Gaudi::Units::GeV;
  const float qop = (std::abs(bdl) < 1.e-8f) ? 0.0f : best_params.qp * qpxz2p;

  // -- Don't make tracks that have grossly too low momentum
  // -- Beware of the momentum resolution!
  const float p = 1.3f * std::abs(1 / qop);
  const float pt = p * std::sqrt(velo_state.tx * velo_state.tx + velo_state.ty * velo_state.ty);

  if (p < PrVeloUTConst::minMomentum || pt < PrVeloUTConst::minPT) return;

  // the track will be added
  uint n_tracks = atomicAdd(n_veloUT_tracks, 1);

  // const float txUT = best_params.xSlopeUTFit;

  // TODO change this to use the pointer to the hits
  // TODO dev_velo_tracks_hits should be const
  const uint velo_track_hit_number = velo_tracks.number_of_hits(i_track);
  const Velo::Consolidated::Hits velo_track_hits = velo_tracks.get_hits(dev_velo_track_hits, i_track);

  // TODO Maybe have a look and optimize this if possible
  // add VELO hits to VeloUT track
  VeloUTTracking::TrackUT track;
  track.hitsNum = 0;
  for (int i=0; i<velo_track_hit_number; ++i) {
    track.addLHCbID(velo_track_hits.LHCbID[i]);
    assert( track.hitsNum < VeloUTTracking::max_track_size);
  }
  track.set_qop( qop );

  // Adding hits to track
  #pragma unroll
  for ( int i = 0; i < N_LAYERS; ++i ) {
    int hit_index = best_hits[i];
    if (hit_index >= 0) {
      track.addLHCbID( ut_hits.LHCbID[hit_index] );
      assert( track.hitsNum < VeloUTTracking::max_track_size);

      // TODO add one overlap hit?
    }
  }
  assert( n_tracks < VeloUTTracking::max_num_tracks );
  VeloUT_tracks[n_tracks] = track;  
}