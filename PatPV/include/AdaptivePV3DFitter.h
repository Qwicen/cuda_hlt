
#ifndef ADAPTIVE_H
#define ADAPTIVE_H

//#include "definitions.h"
#include "AdaptivePVTrack.h"
#include "../../cuda/velo/common/include/VeloDefinitions.cuh"
#include "../../cuda/patPV/include/patPV_Definitions.cuh"






  // Fitting
  bool fitVertex( XYZPoint& seedPoint,
              VeloState * host_velo_states,
             Vertex& vtx,
             std::vector<VeloState>& tracks2remove, int number_of_tracks) ;




  // Get Tukey's weight
  double getTukeyWeight(double trchi2, int iter) ;


#endif ADAPTIVE_H