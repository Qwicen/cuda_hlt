
#ifndef ADAPTIVE_H
#define ADAPTIVE_H

#include "definitions.h"
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



struct Vector2 {
  double x;
  double y;

  Vector2(double m_x, double m_y) : x(m_x), y(m_y){}
};





  class AdaptivePVTrack 
  {
  public:
    AdaptivePVTrack( Track& track, XYZPoint& vtx) ;
    void updateCache( const XYZPoint& vtx ) ;
    double weight() const { return m_weight ; }
    void setWeight(double w) { m_weight = w ;}
    const double *  halfD2Chi2DX2() const { return m_halfD2Chi2DX2 ; }
    const XYZPoint&  halfDChi2DX() const { return m_halfDChi2DX ; }
    double chi2() const { return m_chi2 ; }
    inline double chi2( const XYZPoint& vtx ) const ;
    Track* track() const { return m_track ; }
  private:
    double m_weight ;
     Track* m_track ;
    State m_state ;
    //express symmetrical amtrices as arrays in in packed representation element m(i,j) (j <= i) is supposed to be in array element  (i * (i + 1)) / 2 + j


    double m_invcov[3] ;
    double m_halfD2Chi2DX2[6] ;
    XYZPoint m_halfDChi2DX{1.,2.,3.}  ;
    double m_chi2 ;
    double m_H[6];
  } ;


 



class Vertex {
  public:
    Vertex() {};
    XYZPoint pos{0.,0.,0.};
    double chi2;
    int ndof;
    double cov[6];
    std::vector<Track*> tracks;
    std::vector<double> weights;
    void setChi2AndDoF(double m_chi2, int m_ndof) {
      chi2 = m_chi2;
      ndof = m_ndof;
    }
    void setPosition(XYZPoint& point) {
      pos.x = point.x;
      pos.y = point.y;
      pos.z = point.z;
    }
    void setCovMatrix(double * m_cov) {
      cov[0] = m_cov[0];
      cov[1] = m_cov[1];
      cov[2] = m_cov[2];
      cov[3] = m_cov[3];
      cov[4] = m_cov[4];
      cov[5] = m_cov[5];
    }

    void clearTracks() {
      tracks.clear();
      weights.clear();
    };
    void addToTracks(Track* track, double weight) {
      tracks.push_back(track);
      weights.push_back(weight);
    };
};


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