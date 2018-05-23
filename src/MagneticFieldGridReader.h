
#pragma once

#include <vector>
#include <string>



struct GridQuadrant ;

namespace LHCb {
  class MagneticFieldGrid ;
}

class MagneticFieldGridReader
{
public:
  MagneticFieldGridReader(IMessageSvc& ) ;

  void readFiles( const std::vector<std::string>& filenames) const ;

  StatusCode readDC06File( const std::string& filename,
         LHCb::MagneticFieldGrid& grid ) const ;

  void fillConstantField( const Gaudi::XYZVector& field ,
        LHCb::MagneticFieldGrid& grid ) const ;
private:
  void fillGridFromQuadrants( GridQuadrant* quadrants,
            LHCb::MagneticFieldGrid& grid ) const ;
  void readQuadrant( const std::string& filename,
         GridQuadrant& quad ) const ;

} ;

