
#ifndef ADAPTIVE_H
#define ADAPTIVE_H

#include "VeloDefinitions.cuh"
#include "patPV_Definitions.cuh"

// Fitting
bool fitVertex(
  XYZPoint& seedPoint,
  VeloState* host_velo_states,
  Vertex& vtx,
  int number_of_tracks,
  bool* tracks2disable,
  bool* tracks2remove);

// Get Tukey's weight
double getTukeyWeight(double trchi2, int iter);

#endif ADAPTIVE_H