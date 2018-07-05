#ifndef VPHIT_H
#define VPHIT_H


class VPHit{
 public: 
  
  VPHit( unsigned int lhcbid, float x, float y, float z, int moduleID){
    m_x = x;
    m_y = y;
    m_z = z;
    m_lhcbid = lhcbid;
    m_moduleID = moduleID;
  }
  void setMCParticleKey( int key){
    m_mcp_key = key;   
  }
  int MCParticleKey(){
    return m_mcp_key;
  }
  float x(){
    return m_x;   
  }
  float y(){
    return m_y;
  }
  float z(){
    return m_z;
  }
  int moduleID(){
    return m_moduleID;
  }
  unsigned int lhcbID(){
    return m_lhcbid;
  }
  
  //check if the current hit is linked to same MCParticle of Other VPHit
  bool isMatched( VPHit & otherVPHit){
    if( m_mcp_key == otherVPHit.m_mcp_key) return true;
    return false;
  }
  
  bool isNoise(){
    return (m_mcp_key == -99999 );
  }
 private:
  unsigned int m_lhcbid;
  float m_x;
  float m_y;
  float m_z;
  int m_moduleID;
  int m_mcp_key;
};

#endif

    
