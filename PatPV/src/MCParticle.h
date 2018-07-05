#ifndef MCParticle_H
#define MCParticle_H
#include "pdg.h"
using namespace std;
class MCParticle{
 private:
  bool m_hasSciFi;
  bool m_hasUT;
  bool m_hasVelo;
  int m_key;
  double m_p;
  double m_pt;
  double m_eta;
  double m_vtx_x;
  double m_vtx_y;
  double m_vtx_z;
  bool m_fromB;
  bool m_fromD;
  bool m_fromS;
  int m_grand_grand_motherID;
  int m_pid;
  int m_nAssocFT;
  int m_nAssocUT;
  int m_nAssocVP;
 public: 
  MCParticle() = default;

  void setNbHitsAssociated( int nVP, int nUT, int nFT){
    m_nAssocFT = nFT;
    m_nAssocUT = nUT;
    m_nAssocVP = nVP;
  }
  
  void setKinematic( double p, double pt, double eta){
    m_p = p;
    m_pt = pt;
    m_eta = eta;    
  };

  void setVertexPosition( double vtx_x, double vtx_y, double vtx_z){
      m_vtx_x = vtx_x;
      m_vtx_y = vtx_y;
      m_vtx_z = vtx_z;
  };

    void setflagsDetectors( bool hasVelo, bool hasUT, bool hasSciFi){
      m_hasVelo = hasVelo;
      m_hasUT = hasUT;
      m_hasSciFi = hasSciFi;
    };

    void setProperty( int pid, int grand_grand_motherID, bool fromB, bool fromD, bool fromS){
      m_pid = pid;
      m_grand_grand_motherID = grand_grand_motherID;
      m_fromB = fromB;
      m_fromD = fromD;
      m_fromS = fromS;
    }


    bool isElectron(){
      return abs(m_pid) == PDG::pidElectron;
    }
    bool isPion(){
      return abs(m_pid) == PDG::pidPion;
    }
    bool isKaon(){
      return abs( m_pid) == PDG::pidKaon;
    }
    bool isMuon(){
      return abs( m_pid) == PDG::pidMuon;
    }
    bool isProton(){
      return abs( m_pid) == PDG::pidProton;
    }

    bool isDeutTritHe3Alpha4(){
      return(  abs( m_pid) == PDG::pidDeuterium ||
	       abs( m_pid) == PDG::pidTritium  ||
	       abs( m_pid) == PDG::He3 ||
	       abs( m_pid) == PDG::Alpha );
    }

    bool isLongReconstructible(){
      return m_hasVelo && m_hasSciFi;
    }

    bool isVeloReconstructible(){
      return m_hasVelo;
    }

    bool isVeloUTReconstructible(){
      return m_hasVelo && m_hasUT;
    }

    bool isDownstreamReconstructible(){
      return m_hasUT && m_hasSciFi;
    }

    bool isDownstreamReconstructible_noVelo(){
      return m_hasUT && m_hasSciFi && !m_hasVelo;
    }
    
    bool fromB(){
      return m_fromB;     
    }
    
    bool eta25(){
      return (m_eta> 2 && m_eta<5);
    }
};

#endif

    
