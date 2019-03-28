#ifndef ALLEN_MUONTILEID_H
#define ALLEN_MUONTILEID_H

#include <string>
#include "MuonBase.h"

class MuonLayout {
private:
  unsigned int m_gridX;
  unsigned int m_gridY;
public:
  MuonLayout():m_gridX(0), m_gridY(0) {}

  MuonLayout(unsigned int x, unsigned int y) {
    m_gridX = x;
    m_gridY = y;
  }

  unsigned int x() {
    return m_gridX;
  }

  unsigned int y() {
    return m_gridY;
  }

};

class MuonTileID {
private:
  unsigned int m_muonid;

public:
  unsigned int station() const {
    return ( m_muonid & MuonBase::MaskStation ) << MuonBase::ShiftStation;
  }

  unsigned int region() const {
    return ( m_muonid & MuonBase::MaskRegion ) << MuonBase::ShiftRegion;
  }

  unsigned int quarter() const {
    return ( m_muonid & MuonBase::MaskQuarter ) << MuonBase::ShiftQuarter;
  }

  MuonLayout layout() const {
    unsigned int xg = ( m_muonid & MuonBase::MaskLayoutX ) << MuonBase::ShiftLayoutX;
    unsigned int yg = ( m_muonid & MuonBase::MaskLayoutY ) << MuonBase::ShiftLayoutY;
    return {xg, yg};
  }

  unsigned int nX() const {
    return ( m_muonid & MuonBase::MaskX ) << MuonBase::ShiftX;
  }

  unsigned int nY() const {
    return ( m_muonid & MuonBase::MaskY ) << MuonBase::ShiftY;
  }

  MuonTileID(unsigned int muonid) {
    m_muonid = muonid;
  }
};
#endif //ALLEN_MUONTILEID_H
