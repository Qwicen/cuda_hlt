#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <vector>
#include "../../velo/common/include/VeloDefinitions.cuh"


typedef float data_t;
const data_t WEIGHT = 3966.94;
const float FACTOR = 1.;

struct hit_t {
  float x;
  float y;
  float z;
  int   tid;
  int module;
  bool used = false;
  
   //compare two hits by track ID
  
  bool operator==(int tid_) {
    return tid == tid_;
  } 
};



struct track_t {
  std::vector< const hit_t * > hits;
  std::vector< int > hit_ids;
  //track od
  int key;

};

struct state_t {
  data_t x = 0;
  data_t y = 0;
  data_t z = 0;
  data_t tx = 0;
  data_t ty = 0;
  
  data_t covXX;
  data_t covYY;
  data_t covXTx;
  data_t covYTy;
  data_t covTxTx;
  data_t covTyTy;
  
  data_t chi2 = 0.;
  data_t chi2_x = 0.;
  data_t chi2_y = 0.;
  int key;
  void linearTransportTo (double new_z) {
    const double dz = new_z - z ;
    const double dz2 = dz*dz ;
    x += dz * tx ;
    y += dz * ty ;
    z = new_z;
    covXX += dz2*covTxTx + 2*dz*covXTx ;
    covXTx += dz*covTxTx ;
    covYY += dz2*covTyTy + 2*dz*covYTy ;
    covYTy += dz*covTyTy ;
  }
};


// auxiliary class for searching of clusters of tracks
struct vtxCluster final {

  double  z = 0;            // z of the cluster
  double  sigsq = 0;        // sigma**2 of the cluster
  double  sigsqmin = 0;     // minimum sigma**2 of the tracks forming cluster
  int     ntracks = 0;      // number of tracks in the cluster
  int     not_merged = 0;   // flag for iterative merging

  vtxCluster() = default;

};

struct XYZPoint {
  double x = 0.;
  double y = 0.;
  double z = 0.;
  XYZPoint(double m_x, double m_y, double m_z) : x(m_x), y(m_y), z(m_z) {}

};


struct State {
  double tx = 0.;
  double ty = 0.;
  double x = 0.;
  double y = 0.;
  double z = 0.;
  double errX2 = 0.;
  double errY2 = 0.;
  /*
  void linearTransportTo (double new_z) {
  const double dz = new_z - z ;
  const double dz2 = dz*dz ;
  x += dz * tx ;
  y += dz * ty ;
  z = new_z;




  } */


};

//typedef std::vector<State> Track;
class Track {
public:
  std::vector<state_t> states;
  state_t firstState() {
    return states.at(0);
  }
  XYZPoint slopes() {
    return XYZPoint(states.at(0).tx, states.at(0).ty, 1.);
  }
  XYZPoint position() {
    return XYZPoint(states.at(0).x, states.at(0).y, states.at(0).z);
  }

};


struct Vector2 {
  double x;
  double y;

  Vector2(double m_x, double m_y) : x(m_x), y(m_y){}
};



 



class Vertex {
  public:
    Vertex() {};
    XYZPoint pos{0.,0.,0.};
    double chi2;
    int ndof;
    double cov[6];
    std::vector<VeloState> tracks;
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
    void addToTracks(VeloState track, double weight) {
      tracks.push_back(track);
      weights.push_back(weight);
    };
};

#endif DEFINITIONS_H