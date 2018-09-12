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

#include "VeloUTDefinitions.cuh"
#include "ForwardDefinitions.cuh"
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


class PrForward {

public:

  PrForward();

  std::vector<ForwardTracking::TrackForward> operator()(
    const std::vector<VeloUTTracking::TrackVeloUT>& inputTracks,
    ForwardTracking::HitsSoAFwd *hits_layers_events
  ) const;

  
private:

  const std::vector<std::string> mlpInputVars {{"nPlanes"}, {"dSlope"}, {"dp"}, {"slope2"}, {"dby"}, {"dbx"}, {"day"}};

 
  // Vectors of selected hits
  mutable ForwardTracking::HitsSoAFwd  m_hits_layers;
 
  ReadMLP_Forward1stLoop m_MLPReader_1st;
  ReadMLP_Forward2ndLoop m_MLPReader_2nd;
 
  void find_forward_tracks(  
    const VeloUTTracking::TrackVeloUT& veloUTTrack,
    std::vector<ForwardTracking::TrackForward>& outputTracks
  ) const;

   

  void selectFullCandidates(std::vector<ForwardTracking::TrackForward>& outputTracks,
                            const float xParams_seed[4],
                            const float yParams_seed[4],
			    FullState state_at_endvelo,
			    PrParameters& pars_cur) const;
};

