
#ifndef ADAPTIVE_H
#define ADAPTIVE_H


#include "../../cuda/velo/common/include/VeloDefinitions.cuh"
#include "../../cuda/patPV/include/patPV_Definitions.cuh"







  // Fitting
  bool fitVertex( XYZPoint& seedPoint,
              VeloState * host_velo_states,
             Vertex& vtx, int number_of_tracks, bool * tracks2disable) ;




  // Get Tukey's weight
  double getTukeyWeight(double trchi2, int iter) ;


#endif ADAPTIVE_H