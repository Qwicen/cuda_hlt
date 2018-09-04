#include "../../../main/include/Common.h"

#include "../../../main/include/Tools.h"


//#include "../../../PatPV/include/PVSeedTool.h"
#include "../../../PatPV/include/AdaptivePV3DFitter.h"
#include "../../../cuda/patPV/include/patPV_Definitions.cuh"
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#include <algorithm>



struct MCVertex {
  double x;
  double y;
  double z;
  int numberTracks;
};