
#ifndef ADAPTIVE_H
#define ADAPTIVE_H

//#include "definitions.h"
#include "AdaptivePVTrack.h"


/*
struct State {
  double tx = 0.;
  double ty = 0.;
  double x = 0.;
  double y = 0.;
  double z = 0.;
  double errX2 = 1.;
  double errY2 = 1.;
};

struct XYZPoint {
  double x = 0.;
  double y = 0.;
  double z = 0.;
  XYZPoint(double m_x, double m_y, double m_z) : x(m_x), y(m_y), z(m_z) {}

};

class Track {
public:
  std::vector<State> states;
  State firstState() {
    return states.at(0);
  }
  XYZPoint slopes() {
    return XYZPoint(states.at(0).tx, states.at(0).ty, 1.);
  }
  XYZPoint position() {
    return XYZPoint(states.at(0).x, states.at(0).y, states.at(0).z);
  }

};




*/










class AdaptivePV3DFitter  {

public:
  // Standard constructor
  AdaptivePV3DFitter();
  // Fitting
  bool fitVertex( XYZPoint& seedPoint,
                        std::vector<Track*>& tracks,
                       Vertex& vtx,
                       std::vector<Track*>& tracks2remove) ;
private:
  size_t m_minTr = 4;
  int    m_Iterations = 20;
  int    m_minIter = 5;
  double m_maxDeltaZ = 0.0005; // unit:: mm
  double m_minTrackWeight = 0.00000001;
  double m_TrackErrorScaleFactor = 1.0;
  double m_maxChi2 = 400.0;
  double m_trackMaxChi2 = 12.;
  double m_trackChi ;     // sqrt of trackMaxChi2
  double m_trackMaxChi2Remove = 25.;
  double m_maxDeltaZCache = 1.; //unit: mm


  // Get Tukey's weight
  double getTukeyWeight(double trchi2, int iter) const;
};

#endif ADAPTIVE_H