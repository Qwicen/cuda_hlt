#ifndef FTHIT_H
#define FTHIT_H

class FTHit{
 private :
  unsigned int m_lhcbid;
  float m_xat0;
  float m_zat0;
  float m_dxdy;
  int m_planeCode;
  int m_zone;
  int m_mcp_key;
 public:
  FTHit( unsigned int lhcbid, float xatyeq0, float zat0, float dxdy, int planeCode, int zone){
    m_lhcbid = lhcbid;
    m_xat0 = xatyeq0;
    m_zat0 = zat0;
    m_dxdy = dxdy;
    m_planeCode = planeCode;
    m_zone = zone;
  }
  void setMCParticleKey( int key){
    m_mcp_key = key;   
  }
  int MCParticleKey(){
    return m_mcp_key;
  }
  float x(){
    return m_xat0; 
  }
  float x( float y){
    return m_xat0 + m_dxdy * y;
  }
  float z(){
    return m_xat0;
  }
  
  //check if the current hit is linked to same MCParticle of Other VPHit
  bool isMatched( FTHit & otherFTHit){
    if( m_mcp_key == otherFTHit.MCParticleKey() ) return true;
    return false;
  }
  bool isNoise(){
    //special flag for noise hits!
    if( m_mcp_key == -99999){
      return true;
    }
    return false;
  }
};
#endif
