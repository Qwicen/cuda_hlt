//This code was extracted from the LHCb experiment's LHCb project at https://gitlab.cern.ch/lhcb/LHCb/blob/master/Det/Magnet/src/MagneticFieldGridReader.h

#pragma once

#include <vector>
#include <string>
#include "MagneticFieldGrid.h"


struct GridQuadrant ;

namespace LHCb {
  class MagneticFieldGrid ;
}

class MagneticFieldGridReader
{
public:
  MagneticFieldGridReader( ) ;


  void readFiles( const std::vector<std::string>& filenames,  LHCb::MagneticFieldGrid& grid ) const ;

  void readDC06File( const std::string& filename,
         LHCb::MagneticFieldGrid& grid ) const ;

  void fillConstantField( const XYZVector& field ,
        LHCb::MagneticFieldGrid& grid ) const ;

    void readQuadrant( const std::string& filename,
         GridQuadrant& quad ) const ;


private:
  void fillGridFromQuadrants( GridQuadrant* quadrants,
            LHCb::MagneticFieldGrid& grid ) const ;




} ;

