#pragma once

#include "Common.h"

#include "Tools.h"

#include "AdaptivePV3DFitter.h"
#include "patPV_Definitions.cuh"
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#include <algorithm>

void checkPVs(
  const std::string& folder_name_MC,
  const bool& fromNtuple,
  uint number_of_files,
  Vertex* rec_vertex,
  int* number_of_vertex);

int run_PatPV_on_CPU(
  VeloState* host_velo_states,
  int* host_accumulated_tracks,
  int* host_number_of_tracks_pinned,
  const int& number_of_events,
  Vertex* outvtxvec,
  uint* number_of_vertex,
  XYZPoint* seeds,
  uint* number_of_seeds

);
