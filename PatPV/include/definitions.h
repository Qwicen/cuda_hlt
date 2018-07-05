#ifndef DEFINITIONS_H
#define DEFINITIONS_H



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
  void linearTransportTo (double new_z) {
  const double dz = new_z - z ;
  const double dz2 = dz*dz ;
  x += dz * tx ;
  y += dz * y ;
  z = new_z;


  }


};

//typedef std::vector<State> Track;
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


#endif DEFINITIONS_H