#include "compassUT.cuh"

#include <float.h>

__global__ void compassUT(
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
  const uint* dev_unique_x_sector_offsets, // TODO remove this, only needed for decoding
  const float* dev_unique_sector_xs, // list of xs that define the groups
  VeloUTTracking::TrackUT* dev_compassUT_tracks,
  int* dev_atomics_compassUT, // size of number of events
  int* dev_windows_layers)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint total_number_of_hits = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks{
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states{dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  UTHitOffsets ut_hit_offsets{
    dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};

  UTHits ut_hits;
  ut_hits.typecast_sorted(dev_ut_hits, total_number_of_hits);

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

  // const float* fudgeFactors = &(dev_ut_magnet_tool->dxLayTable[0]);
  // const float* bdlTable     = &(dev_ut_magnet_tool->bdlTable[0]);

  // array to store indices of selected hits in layers
  // -> can then access the hit information in the HitsSoA
  // int hitCandidatesInLayers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];
  // int n_hitCandidatesInLayers[VeloUTTracking::n_layers];

  for (int i_track = threadIdx.x; i_track < number_of_tracks_event; i_track += blockDim.x) {

    // __syncthreads();

    // TODO the non active tracks should be -1
    // const int i_track = shared_active_tracks[threadIdx.x];

    const uint velo_states_index = event_tracks_offset + i_track;
    const MiniState velo_state{velo_states, velo_states_index};

    //   __syncthreads();

    // TrackHelper helper{velo_state};

    // indices within hitCandidatesInLayers for selected hits belonging to best track
    // int hitCandidateIndices[N_LAYERS];

    // TODO remove the x_hit_layer (not needed)
    float x_hit_layer[N_LAYERS];
    int best_hits[N_LAYERS] = {-1, -1, -1, -1};
    BestParams best_params;

    // TODO search backwards if we didn't found 4 hits
    find_best_hits(
      i_track,
      dev_windows_layers,
      ut_hits,
      ut_hit_offsets,
      velo_state,
      dev_ut_dxDy,
      true,
      x_hit_layer,
      best_hits,
      best_params);

    // Count best hits
    int total_num_hits = 0;
    for (int i = 0; i < N_LAYERS; ++i) {
      if (best_hits[i] >= 0) total_num_hits++;
    }

    const float* bdl_table = &(dev_ut_magnet_tool->bdlTable[0]);

    // write the final track
    if (total_num_hits > 0) {
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

    //     if ( helper.n_hits > 0 ) {
    //       const uint velo_track_hit_number = velo_tracks.number_of_hits(i_track);
    //       const Velo::Consolidated::Hits velo_track_hits = velo_tracks.get_hits(dev_velo_track_hits, i_track);

    //       prepareOutputTrack(
    //         i_track,
    //         velo_track_hits,
    //         velo_track_hit_number,
    //         helper,
    //         velo_state,
    //         (int*) &windows_layers[0],
    //         ut_hits,
    //         ut_hit_count,
    //         (float*) &x_hit_layer[0],
    //         (int*) &hitCandidateIndices[0],
    //         bdlTable,
    //         veloUT_tracks_event,
    //         n_veloUT_tracks_event);
    //     }

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

  //   // MiniState aux_velo_state {velo_states, velo_states_index};
  //   const uint velo_states_index = event_tracks_offset + i_track;
  //   const MiniState velo_state {velo_states, velo_states_index};

  //   get_windows(
  //     i_track,
  //     velo_state,
  //     fudgeFactors,
  //     ut_hits,
  //     ut_hit_count,
  //     dev_ut_dxDy,
  //     (int*) &windows_layers[0]);

  //   __syncthreads();

  //   // for storing calculated x position of hits for this track
  //   // float x_pos_layers[VeloUTTracking::n_layers][VeloUTTracking::max_hit_candidates_per_layer];

  // }
}

//=========================================================================
// hits_to_track
//=========================================================================
__host__ __device__ void find_best_hits(
  const int i_track,
  const int* dev_windows_layers,
  const UTHits& ut_hits,
  const UTHitOffsets& ut_hit_count,
  const MiniState& velo_state,
  const float* ut_dxDy,
  const bool forward,
  float* x_hit_layer,
  int* best_hits,
  BestParams& best_params)
{
  // handle forward / backward cluster search
  int layers[N_LAYERS];
  for (int i_layer = 0; i_layer < N_LAYERS; ++i_layer) {
    if (forward)
      layers[i_layer] = i_layer;
    else
      layers[i_layer] = N_LAYERS - 1 - i_layer;
  }

  // Get windows of all layers
  WindowIndicator win_ranges(dev_windows_layers); 
  const auto* ranges = win_ranges.get_track_candidates(i_track);
  const int from0 = ranges->layer[0].first;
  const int to0 = ranges->layer[0].last;
  const int from2 = ranges->layer[2].first;
  const int to2 = ranges->layer[2].last;
  const int from1 = ranges->layer[1].first;
  const int to1 = ranges->layer[1].last;
  const int from3 = ranges->layer[3].first;
  const int to3 = ranges->layer[3].last;
  
  // printf("from0: %i, to0: %i, from1: %i, to1: %i, from2: %i, to2: %i, from3: %i, to3: %i\n", from0, to0, from1, to1, from2, to2, from3, to3);

  const float yyProto = velo_state.y - velo_state.ty * velo_state.z;

  // auto is_valid = [](float dx, int layer, float y){
  //   if (dx < )
  // }

  // const float normFactNum = normFact[layer];
  // const float invNormFact = 1.0/normFactNum;
  // xTol*invNormFact
  // const auto zInit = ut_hits.zAtYEq0[layer_offset + posBeg];
  // const auto xOnTrackProto = myState.x + myState.tx*(zInit - myState.z);
  // const auto yApprox = myState.y + myState.ty * (zInit - myState.z);
  // const auto xx = ut_hits.xAt(layer_offset + i, yApprox, dxDy); 
  // const auto dx = xx - xOnTrackProto;
  
  // if( dx < -xTolNormFact ) continue;
  // if( dx >  xTolNormFact ) continue; 
  
  // // -- Now refine the tolerance in Y
  // if ( ut_hits.isNotYCompatible( layer_offset + i, yApprox, PrVeloUTConst::yTol + PrVeloUTConst::yTolSlope * std::abs(dx*invNormFact)) ) continue;

  // BestParams best_params;

  for (int i_hit0 = from0; i_hit0 < to0; ++i_hit0) {

    const float yy0 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit0]);
    x_hit_layer[0] = ut_hits.xAt(i_hit0, yy0, ut_dxDy[layers[0]]);
    const float zhitLayer0 = ut_hits.zAtYEq0[i_hit0];
    best_hits[0] = i_hit0;

    for (int i_hit2 = from2; i_hit2 < to2; ++i_hit2) {

      const float yy2 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit2]);
      x_hit_layer[2] = ut_hits.xAt(i_hit2, yy2, ut_dxDy[layers[2]]);
      const float zhitLayer2 = ut_hits.zAtYEq0[i_hit2];
      best_hits[2] = i_hit2;

      // same bool check for the hit
      const float tx = (x_hit_layer[2] - x_hit_layer[0]) / (zhitLayer2 - zhitLayer0);
      if (std::abs(tx - velo_state.tx) <= PrVeloUTConst::deltaTx2) {
        float hitTol = PrVeloUTConst::hitTol2;

        // Search for triplet
        for (int i_hit1 = from1; i_hit1 < to1; ++i_hit1) {
          const float yy1 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit1]);
          x_hit_layer[1] = ut_hits.xAt(i_hit1, yy1, ut_dxDy[layers[1]]);
          const float zhitLayer1 = ut_hits.zAtYEq0[i_hit1];
          const float xextrapLayer1 = x_hit_layer[0] + tx * (zhitLayer1 - zhitLayer0);
          if (std::abs(x_hit_layer[1] - xextrapLayer1) < hitTol) {
            hitTol = std::abs(x_hit_layer[1] - xextrapLayer1);
            // index_best_hit_1 = i_hit1;
            best_hits[1] = i_hit1;
          }
        }

        // Search for cuadruplet
        hitTol = PrVeloUTConst::hitTol2;
        for (int i_hit3 = from3; i_hit3 < to3; ++i_hit3) {
          const float yy3 = yyProto + (velo_state.ty * ut_hits.zAtYEq0[i_hit3]);
          x_hit_layer[3] = ut_hits.xAt(i_hit3, yy3, ut_dxDy[layers[3]]);
          const float zhitLayer3 = ut_hits.zAtYEq0[i_hit3];
          const float xextrapLayer3 = x_hit_layer[2] + tx * (zhitLayer3 - zhitLayer2);
          if (std::abs(x_hit_layer[3] - xextrapLayer3) < hitTol) {
            hitTol = std::abs(x_hit_layer[3] - xextrapLayer3);
            // index_best_hit_3 = i_hit3;
            best_hits[3] = i_hit3;
          }
        }

        // Fit the hits to get q/p, chi2
        best_params = pkick_fit(best_hits, ut_hits, velo_state, ut_dxDy, yyProto);
      }
    }
  }

  if (best_hits[0] != -1 && best_hits[1] != -1 && best_hits[2] != -1 && best_hits[3] != -1) {
    printf("hit0: %i, hit1: %i, hit2: %i, hit3: %i, q/p: %f, chi2: %f\n", best_hits[0], best_hits[1], best_hits[2], best_hits[3], best_params.qp, best_params.chi2UT);
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
  for (int i = 0; i < N_LAYERS; ++i) {
    if (best_hits[i] >= 0) { n_high_thres += ut_hits.highThreshold[best_hits[i]]; }
  }

  // Veto hit combinations with no high threshold hit
  // = likely spillover
  if (n_high_thres < PrVeloUTConst::minHighThres) return best_params;

  // Scale the z-component, to not run into numerical problems with floats
  // first add to sum values from hit at xMidField, zMidField hit
  const float zDiff = 0.001 * (PrVeloUTConst::zKink - PrVeloUTConst::zMidUT);

  // Helper stuff from velo state
  const float xMidField = velo_state.x + velo_state.tx * (PrVeloUTConst::zKink - velo_state.z);
  const float a = PrVeloUTConst::sigmaVeloSlope * (PrVeloUTConst::zKink - velo_state.z);
  const float wb = 1. / (a * a);

  float mat[3] = {wb, wb * zDiff, wb * zDiff * zDiff};
  float rhs[2] = {wb * xMidField, wb * xMidField * zDiff};

  // add hits
  for (int i = 0; i < N_LAYERS; ++i) {
    int hit_index = best_hits[i];
    if (hit_index >= 0) {

      const float wi = ut_hits.weight[hit_index];
      const float dxDy = ut_dxDy[ut_hits.planeCode[hit_index]];
      const float ci = ut_hits.cosT(hit_index, dxDy);
      const float dz = 0.001 * (ut_hits.zAtYEq0[hit_index] - PrVeloUTConst::zMidUT);
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

  const float denom = 1. / (mat[0] * mat[2] - mat[1] * mat[1]);
  const float xSlopeUTFit = 0.001 * (mat[0] * rhs[1] - mat[1] * rhs[0]) * denom;
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
    const float sinInX = xSlopeVeloFit * std::sqrt(1. + xSlopeVeloFit * xSlopeVeloFit);
    const float sinOutX = xSlopeUTFit * std::sqrt(1. + xSlopeUTFit * xSlopeUTFit);

    best_params.qp = sinInX - sinOutX;
    best_params.chi2UT = chi2UT;
    best_params.xUTFit = xUTFit;
    best_params.xSlopeUTFit = xSlopeUTFit;
  }

  return best_params;
}

// These things are all hardcopied from the PrTableForFunction and PrUTMagnetTool
// If the granularity or whatever changes, this will give wrong results
__host__ __device__ int master_index(const int index1, const int index2, const int index3){
  return (index3*11 + index2)*31 + index1;
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
  const float zOrigin = (std::fabs(velo_state.ty) > 0.001) ? velo_state.z - velo_state.y / velo_state.ty
                                                           : velo_state.z - velo_state.x / velo_state.tx;

  // -- These are calculations, copied and simplified from PrTableForFunction
  const float var[3] = {velo_state.ty, zOrigin, velo_state.z};

  const int index1 = std::max(0, std::min(30, int((var[0] + 0.3) / 0.6 * 30)));
  const int index2 = std::max(0, std::min(10, int((var[1] + 250) / 500 * 10)));
  const int index3 = std::max(0, std::min(10, int(var[2] / 800 * 10)));

  assert(master_index(index1, index2, index3) < PrUTMagnetTool::N_bdl_vals);
  float bdl = bdl_table[master_index(index1, index2, index3)];

  const int num_idx = 3;
  const float bdls[num_idx] = {bdl_table[master_index(index1 + 1, index2, index3)],
                               bdl_table[master_index(index1, index2 + 1, index3)],
                               bdl_table[master_index(index1, index2, index3 + 1)]};
  const float deltaBdl[num_idx] = {0.02, 50.0, 80.0};
  const float boundaries[num_idx] = {
    -0.3f + float(index1) * deltaBdl[0], -250.0f + float(index2) * deltaBdl[1], 0.0f + float(index3) * deltaBdl[2]};

  // This is an interpolation, to get a bit more precision
  float addBdlVal = 0.0;
  const float minValsBdl[num_idx] = {-0.3, -250.0, 0.0};
  const float maxValsBdl[num_idx] = {0.3, 250.0, 800.0};
  for (int i = 0; i < num_idx; ++i) {
    if (var[i] < minValsBdl[i] || var[i] > maxValsBdl[i]) continue;
    const float dTab_dVar = (bdls[i] - bdl) / deltaBdl[i];
    const float dVar = (var[i] - boundaries[i]);
    addBdlVal += dTab_dVar * dVar;
  }
  bdl += addBdlVal;

  const float qpxz2p = -1 * std::sqrt(1. + velo_state.ty * velo_state.ty) / bdl * 3.3356 / Gaudi::Units::GeV;
  const float qop = (std::abs(bdl) < 1.e-8) ? 0.0 : best_params.qp * qpxz2p;

  // -- Don't make tracks that have grossly too low momentum
  // -- Beware of the momentum resolution!
  const float p = 1.3 * std::abs(1 / qop);
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

  // const float yyProto = velo_state.y - velo_state.ty * velo_state.z;

  // Adding overlap hits
  for ( int i = 0; i < num_best_hits; ++i ) {
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