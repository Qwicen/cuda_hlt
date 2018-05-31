#ifndef PRKERNEL_UTHITINFO_H
#define PRKERNEL_UTHITINFO_H 1
// Include files
#include "Event/STLiteCluster.h"
#include "DeSTDetector.h"
#include "DeSTSector.h"
#include <string>
/** @class PrUTHitInfo PrUTHitInfo.h PrKernel/PrUTHitInfo.h
 *  @author Renato Quagliani, Christoph Hasse
 *  @date   2016-11-15
 */
namespace UT
{
  namespace Info
  {
    //--- Final Hit Location
    const std::string HitLocation = "UT/UTHits";
    //--- Detector Location of UT
    const std::string DetLocation = DeSTDetLocation::UT;
    // DeSTDetLocation::location("UT");
    //--- Cluster location
    const std::string ClusLocation = LHCb::STLiteClusterLocation::UTClusters;

    const int kMinStation = 0;  ///< Minimum valid station number for UT
    const int kMaxStation = 1;  ///< Maximum valid station number for UT
    const int kNStations  = 2;  ///< Number of UT stations
    const int kMinLayer   = 0;  ///< Minimum valid layer number for a UT station
    const int kMaxLayer   = 1;  ///< Maximum valid layer number for a UT station
    const int kNLayers    = 2;  ///< Number of UT layers within a station
    const int kMinRegion  = 0;  ///< Minimum valid region number for a UT layer
    const int kMaxRegion  = 11; ///< Maximum valid region number for a UT layer
    const int kNRegions   = 12; ///< Number of UT regions within a layer
  }
}
#endif // PRKERNEL_UTHITINFO_H
