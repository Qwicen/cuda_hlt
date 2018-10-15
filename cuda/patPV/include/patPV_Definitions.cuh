#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <vector>
#include "VeloDefinitions.cuh"

namespace PatPV {


//maximum number of vertices in a event
static constexpr uint max_number_vertices = 30;


// auxiliary class for searching of clusters of tracks



}


struct vtxCluster final {

  double  z = 0;            // z of the cluster
  double  sigsq = 0;        // sigma**2 of the cluster
  double  sigsqmin = 0;     // minimum sigma**2 of the tracks forming cluster
  int     ntracks = 1;      // number of tracks in the cluster
  bool     merged = false;   // flag for iterative merging

  vtxCluster() = default;

};

__device__ struct XYZPoint {
  double x = 0.;
  double y = 0.;
  double z = 0.;
  __device__ XYZPoint(double m_x, double m_y, double m_z) : x(m_x), y(m_y), z(m_z) {};
  __device__ XYZPoint() {};

};






__device__ struct Vector2 {
  double x;
  double y;

  __device__ Vector2(double m_x, double m_y) : x(m_x), y(m_y){}
};

 



__device__ class Vertex {
  public:
    __device__ Vertex() {};
    double x = 0.;
    double y = 0.;
    double z = 0.;
    double chi2;
    int ndof;

    double cov00 = 0.;
    double cov10 = 0.;
    double cov11 = 0.;
    double cov20 = 0.;
    double cov21 = 0.;
    double cov22 = 0.;


    __device__ void setChi2AndDoF(double m_chi2, int m_ndof) {
      chi2 = m_chi2;
      ndof = m_ndof;
    }
    __device__ void setPosition(XYZPoint& point) {
      x = point.x;
      y = point.y;
      z = point.z;
    }
    __device__ void setCovMatrix(double * m_cov) {
      cov00 = m_cov[0];
      cov10 = m_cov[1];
      cov11 = m_cov[2];
      cov20 = m_cov[3];
      cov21 = m_cov[4];
      cov22 = m_cov[5];
    }
   int nTracks = 0.;


};

#endif DEFINITIONS_H