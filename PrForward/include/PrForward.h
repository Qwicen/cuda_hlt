#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

#include "Logger.h"

#include "TMVA_MLP_Forward1stLoop.h"
#include "TMVA_MLP_Forward2ndLoop.h"
#include "SystemOfUnits.h"

#include "VeloDefinitions.cuh"
#include "VeloUTDefinitions.cuh"
#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.h"
#include "TrackUtils.h"
#include "LinearFitting.h"
#include "HitUtils.h"
#include "HoughTransform.h"
#include "FindXHits.h"
#include "FindStereoHits.h"

/** @class PrForward PrForward.h
   *
   *  - InputTracksName: Input location for VeloUT tracks
   *  - OutputTracksName: Output location for Forward tracks
   *  Based on code written by
   *  2012-03-20 : Olivier Callot
   *  2013-03-15 : Thomas Nikodem
   *  2015-02-13 : Sevda Esen [additional search in the triangles by Marian Stahl]
   *  2016-03-09 : Thomas Nikodem [complete restructuring]
   */


std::vector<SciFi::Track> PrForward(
  const std::vector<VeloUTTracking::TrackUT>& inputTracks,
  SciFi::HitsSoA *hits_layers_events,
  const VeloState * velo_states,
  const int velo_number_of_tracks);
                                                      
void find_forward_tracks(
  SciFi::HitsSoA* hits_layers,  
  const VeloUTTracking::TrackUT& veloUTTrack,
  std::vector<SciFi::Track>& outputTracks,
  const ReadMLP_Forward1stLoop& MLPReader_1st,
  const ReadMLP_Forward2ndLoop& MLPReader_2nd,
  const VeloState& velo_state);


void selectFullCandidates(
  SciFi::HitsSoA* hits_layers,
  std::vector<SciFi::Track>& outputTracks,
  const float xParams_seed[4],
  const float yParams_seed[4],
  VeloState velo_state,
  const float VeloUT_qOverP,
  SciFi::Tracking::HitSearchCuts& pars_cur,
  const ReadMLP_Forward1stLoop& MLPReader_1st,
  const ReadMLP_Forward2ndLoop& MLPReader_2nd);

