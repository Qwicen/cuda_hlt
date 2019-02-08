#include "PrForwardTools.cuh"

/* Look first in x layers, then in stereo layers for hits
   do 1D Hough transform for x- and stereo hits
   do global 1D Hough transform
   use TMVAs to obtain track quality */
__host__ __device__ void find_forward_tracks(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  const float qop_ut,
  const int i_veloUT_track,
  SciFi::TrackHits* outputTracks,
  uint* n_forward_tracks,
  const SciFi::Tracking::TMVA* tmva1,
  const SciFi::Tracking::TMVA* tmva2,
  const SciFi::Tracking::Arrays* constArrays,
  const MiniState& velo_state)
{

  // The LHCb framework code had a PT preselection for the VeloUT tracks
  // here, which I am removing because this should be done explicitly through
  // track selectors if we do it at all, not hacked inside the tracking code

  const float zRef_track = SciFi::Tracking::zReference;
  const float xAtRef = xFromVelo(zRef_track, velo_state);
  const float xParams_seed[4] = {xAtRef, velo_state.tx, 0.f, 0.f};
  const float yAtRef = yFromVelo(zRef_track, velo_state);
  const float yParams_seed[4] = {yAtRef, velo_state.ty, 0.f, 0.f};

  // First loop Hough cluster search, set initial search windows
  SciFi::Tracking::HitSearchCuts pars_first {SciFi::Tracking::minXHits,
                                             SciFi::Tracking::maxXWindow,
                                             SciFi::Tracking::maxXWindowSlope,
                                             SciFi::Tracking::maxXGap,
                                             4u};
  SciFi::Tracking::HitSearchCuts pars_second {SciFi::Tracking::minXHits_2nd,
                                              SciFi::Tracking::maxXWindow_2nd,
                                              SciFi::Tracking::maxXWindowSlope_2nd,
                                              SciFi::Tracking::maxXGap_2nd,
                                              4u};

  int allXHits[2][SciFi::Tracking::max_x_hits];
  int n_x_hits[2] = {0};
  float coordX[2][SciFi::Tracking::max_x_hits];

  if (yAtRef > -5.f)
    collectAllXHits(
      scifi_hits,
      scifi_hit_count,
      allXHits[1],
      n_x_hits[1],
      coordX[1],
      xParams_seed,
      yParams_seed,
      constArrays,
      velo_state,
      qop_ut,
      1);
  if (yAtRef < 5.f)
    collectAllXHits(
      scifi_hits,
      scifi_hit_count,
      allXHits[0],
      n_x_hits[0],
      coordX[0],
      xParams_seed,
      yParams_seed,
      constArrays,
      velo_state,
      qop_ut,
      -1);

  SciFi::Tracking::Track candidate_tracks[SciFi::Tracking::max_candidate_tracks];
  int n_candidate_tracks = 0;
  bool usedHits[2][SciFi::Tracking::max_x_hits] = {false};

  if (yAtRef > -5.f)
    selectXCandidates(
      scifi_hits,
      scifi_hit_count,
      allXHits[1],
      n_x_hits[1],
      usedHits[1],
      coordX[1],
      candidate_tracks,
      n_candidate_tracks,
      zRef_track,
      xParams_seed,
      yParams_seed,
      velo_state,
      pars_first,
      constArrays,
      1,
      false);
  if (yAtRef < 5.f)
    selectXCandidates(
      scifi_hits,
      scifi_hit_count,
      allXHits[0],
      n_x_hits[0],
      usedHits[0],
      coordX[0],
      candidate_tracks,
      n_candidate_tracks,
      zRef_track,
      xParams_seed,
      yParams_seed,
      velo_state,
      pars_first,
      constArrays,
      -1,
      false);

  SciFi::Tracking::Track selected_tracks[SciFi::Tracking::max_selected_tracks];
  int n_selected_tracks = 0;

  selectFullCandidates(
    scifi_hits,
    scifi_hit_count,
    candidate_tracks,
    n_candidate_tracks,
    selected_tracks,
    n_selected_tracks,
    xParams_seed,
    yParams_seed,
    velo_state,
    qop_ut,
    pars_first,
    tmva1,
    tmva2,
    constArrays,
    false);

  bool ok = false;
  for (int i_track = 0; i_track < n_selected_tracks; ++i_track) {
    if (selected_tracks[i_track].hitsNum > 10) ok = true;
  }
  assert(n_selected_tracks < SciFi::Tracking::max_selected_tracks);

  SciFi::Tracking::Track candidate_tracks2[SciFi::Tracking::max_tracks_second_loop];
  int n_candidate_tracks2 = 0;

  if (!ok && SciFi::Tracking::secondLoop) { // If you found nothing begin the 2nd loop
    if (yAtRef > -5.f)
      selectXCandidates(
        scifi_hits,
        scifi_hit_count,
        allXHits[1],
        n_x_hits[1],
        usedHits[1],
        coordX[1],
        candidate_tracks2,
        n_candidate_tracks2,
        zRef_track,
        xParams_seed,
        yParams_seed,
        velo_state,
        pars_second,
        constArrays,
        1,
        true);
    if (yAtRef < 5.f)
      selectXCandidates(
        scifi_hits,
        scifi_hit_count,
        allXHits[0],
        n_x_hits[0],
        usedHits[0],
        coordX[0],
        candidate_tracks2,
        n_candidate_tracks2,
        zRef_track,
        xParams_seed,
        yParams_seed,
        velo_state,
        pars_second,
        constArrays,
        -1,
        true);

    SciFi::Tracking::Track selected_tracks2[SciFi::Tracking::max_tracks_second_loop];
    int n_selected_tracks2 = 0;

    selectFullCandidates(
      scifi_hits,
      scifi_hit_count,
      candidate_tracks2,
      n_candidate_tracks2,
      selected_tracks2,
      n_selected_tracks2,
      xParams_seed,
      yParams_seed,
      velo_state,
      qop_ut,
      pars_second,
      tmva1,
      tmva2,
      constArrays,
      true);

    for (int i_track = 0; i_track < n_selected_tracks2; ++i_track) {
      assert(n_selected_tracks < SciFi::Tracking::max_selected_tracks);
      selected_tracks[n_selected_tracks++] = selected_tracks2[i_track];
    }

    ok = (n_selected_tracks > 0);
  }

  if (ok || !SciFi::Tracking::secondLoop) {

    if (n_selected_tracks > 1) {
      // not using thrust::sort due to temporary_buffer::allocate:: get_temporary_buffer failed" error
      // thrust::sort( thrust::seq, selected_tracks, selected_tracks + n_selected_tracks, lowerByQuality);
      sort_tracks(selected_tracks, n_selected_tracks, [](SciFi::Tracking::Track t1, SciFi::Tracking::Track t2) {
        if (t1.quality < t2.quality) return -1;
        if (t1.quality == t2.quality) return 0;
        return 1;
      });
    }

    const uint event_hit_offset = scifi_hit_count.event_offset();
    float minQuality = SciFi::Tracking::maxQuality;
    for (int i_track = 0; i_track < n_selected_tracks; ++i_track) {
      SciFi::Tracking::Track& track = selected_tracks[i_track];
      if (track.quality + SciFi::Tracking::deltaQuality < minQuality)
        minQuality = track.quality + SciFi::Tracking::deltaQuality;
      if (!(track.quality > minQuality)) {

        SciFi::TrackHits tr = makeTrack(track);
        tr.UTTrackIndex = i_veloUT_track;

        // add LHCbIDs from SciFi part of the track
        for (int i_hit = 0; i_hit < track.hitsNum; ++i_hit) {
          // save local hit index within event to be able to use short
          const int local_hit_index = track.hit_indices[i_hit] - event_hit_offset;
          tr.addHit(local_hit_index);
        }
        assert(tr.hitsNum < SciFi::Constants::max_track_size);

        if (*n_forward_tracks >= SciFi::Constants::max_tracks) printf("n_forward_tracks = %u \n", *n_forward_tracks);
        assert(*n_forward_tracks < SciFi::Constants::max_tracks);
#ifndef __CUDA_ARCH__
        outputTracks[(*n_forward_tracks)++] = tr;
#else
        uint n_tracks = atomicAdd(n_forward_tracks, 1);
        assert(n_tracks < SciFi::Constants::max_tracks);
        outputTracks[n_tracks] = tr;
#endif
      }
    }
  }
}

// Turn SciFi::Tracking::Track into a SciFi::Track
__host__ __device__ SciFi::TrackHits makeTrack(SciFi::Tracking::Track track)
{
  SciFi::TrackHits tr;
  tr.qop = track.qop;
  tr.chi2 = track.chi2;

  // add state at zEndT
  const float z = SciFi::Constants::ZEndT;
  MiniState state(track.x(z), track.y(z), z, track.xSlope(z), track.ySlope(z));

  tr.state = state;

  return tr;
}

//=========================================================================
//  Create Full candidates out of xCandidates
//  Searching for stereo hits
//  Fit of all hits
//  save everything in track candidate folder
//=========================================================================
__host__ __device__ void selectFullCandidates(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  SciFi::Tracking::Track* candidate_tracks,
  int& n_candidate_tracks,
  SciFi::Tracking::Track* selected_tracks,
  int& n_selected_tracks,
  const float xParams_seed[4],
  const float yParams_seed[4],
  MiniState velo_state,
  const float VeloUT_qOverP,
  SciFi::Tracking::HitSearchCuts& pars,
  const SciFi::Tracking::TMVA* tmva1,
  const SciFi::Tracking::TMVA* tmva2,
  const SciFi::Tracking::Arrays* constArrays,
  const bool secondLoop)
{

  PlaneCounter planeCounter;
  planeCounter.clear();
  float mlpInput[7] = {0};

  for (int i_track = 0; i_track < n_candidate_tracks; ++i_track) {
    SciFi::Tracking::Track* cand = candidate_tracks + i_track;

    pars.minStereoHits = 4;

    if (cand->hitsNum + pars.minStereoHits < SciFi::Tracking::minTotalHits) {
      pars.minStereoHits = SciFi::Tracking::minTotalHits - cand->hitsNum;
    }

    int stereoHits[SciFi::Tracking::max_stereo_hits];
    int n_stereoHits = 0;
    float stereoCoords[SciFi::Tracking::max_stereo_hits];
    collectStereoHits(
      scifi_hits, scifi_hit_count, *cand, velo_state, pars, constArrays, stereoCoords, stereoHits, n_stereoHits);

    if (n_stereoHits < pars.minStereoHits) continue;

    if (!selectStereoHits(
          scifi_hits, scifi_hit_count, *cand, constArrays, stereoCoords, stereoHits, n_stereoHits, velo_state, pars))
      continue;

    planeCounter.clear();
    for (int i_hit = 0; i_hit < cand->hitsNum; ++i_hit) {
      int hit = cand->hit_indices[i_hit];
      planeCounter.addHit(scifi_hits.planeCode(hit) / 2);
    }

    // make a fit of ALL hits using their x coordinate
    if (!quadraticFitX(scifi_hits, cand->trackParams, cand->hit_indices, cand->hitsNum, planeCounter, pars)) continue;

    // track has enough hits, calcualte quality and save if good enough
    if (planeCounter.nbDifferent >= SciFi::Tracking::minTotalHits) {

      const float qOverP = calcqOverP(cand->trackParams[1], constArrays, velo_state);
      // orig params before fitting , TODO faster if only calc once?? mem usage?
      const float xAtRef = cand->trackParams[0];
      float dSlope = (velo_state.x + (SciFi::Tracking::zReference - velo_state.z) * velo_state.tx - xAtRef) /
                     (SciFi::Tracking::zReference - constArrays->zMagnetParams[0]);
      const float zMagSlope =
        constArrays->zMagnetParams[2] * pow(velo_state.tx, 2) + constArrays->zMagnetParams[3] * pow(velo_state.ty, 2);
      const float zMag = constArrays->zMagnetParams[0] + constArrays->zMagnetParams[1] * dSlope * dSlope + zMagSlope;
      const float xMag = velo_state.x + (zMag - velo_state.z) * velo_state.tx;
      const float slopeT = (xAtRef - xMag) / (SciFi::Tracking::zReference - zMag);
      dSlope = slopeT - velo_state.tx;
      const float dyCoef = dSlope * dSlope * velo_state.ty;

      float bx = slopeT;
      float ay = velo_state.y + (SciFi::Tracking::zReference - velo_state.z) * velo_state.ty;
      float by = velo_state.ty + dyCoef * SciFi::Tracking::byParams;

      // ay,by,bx params
      const float ay1 = cand->trackParams[4];
      const float by1 = cand->trackParams[5];
      const float bx1 = cand->trackParams[1];

      mlpInput[0] = planeCounter.nbDifferent;
      mlpInput[1] = qOverP;
      mlpInput[2] = VeloUT_qOverP - qOverP;                // veloUT - scifi
      if (fabsf(VeloUT_qOverP) < 1e-9f) mlpInput[2] = 0.f; // no momentum estiamte
      mlpInput[3] = pow(velo_state.tx, 2) + pow(velo_state.ty, 2);
      mlpInput[4] = by - by1;
      mlpInput[5] = bx - bx1;
      mlpInput[6] = ay - ay1;

      float quality = 0.f;
      /// WARNING: if the NN classes straight out of TMVA are used, put a mutex here!
      if (pars.minXHits > 4)
        quality = GetMvaValue(mlpInput, tmva1); // 1st loop NN
      else
        quality = GetMvaValue(mlpInput, tmva2); // 2nd loop NN

      quality = 1.f - quality; // backward compability

      if (quality < SciFi::Tracking::maxQuality) {
        cand->quality = quality;
        cand->qop = qOverP;
        if (!secondLoop)
          assert(n_selected_tracks < SciFi::Tracking::max_selected_tracks);
        else if (secondLoop)
          assert(n_selected_tracks < SciFi::Tracking::max_tracks_second_loop);
        selected_tracks[n_selected_tracks++] = *cand;
        if (!secondLoop) {
          if (n_selected_tracks >= SciFi::Tracking::max_selected_tracks) break;
        }
        else if (secondLoop) {
          if (n_selected_tracks >= SciFi::Tracking::max_tracks_second_loop) break;
        }
      }
    }
  }
}
