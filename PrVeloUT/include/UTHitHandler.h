#ifndef PRKERNEL_UTHITHANDLER_H
#define PRKERNEL_UTHITHANDLER_H 1

// Include files
#include "../include/DataObject.h"
#include "../include/ObjectContainerBase.h"
#include "../include/Range.h"
#include "../include/UTHitInfo.h"
#include "../include/UTHit.h"
#include "../include/IndexedHitContainer.h"
#include "../include/MultiIndexedHitContainer.h"

/** @class UTHitHandler UTHitHandler.h PrKernel/UTHitHandler.h
 *
 *  UTHitHandler contains the hits in the UT detector and the accessor to them
 *  TODO: convert it to a 2D IndexedHitContainer.
 *  @author Renato Quagliani, Christoph Hasse
 *  @date   2016-11-16
 */
namespace UT
{
  class HitHandler
  {
  public:
    /// Standard constructor
    using HitContainer = MultiIndexedHitContainer<UT::Hit, UT::Info::kNStations, UT::Info::kNLayers>;

    using HitRange = typename HitContainer::HitRange;

    HitHandler()                              = default;
    HitHandler( const HitHandler& other ) = default;
    HitHandler( HitHandler&& other )      = default;
    HitHandler& operator=( const HitHandler& other ) = default;
    HitHandler& operator=( HitHandler&& other ) = default;

    // Constructor with the capacity of the HitContainer
    HitHandler(int size) : m_hits(size) {};

    // Method to add Hit in the container
    void AddHit( const DeSTSector* aSector, const LHCb::STLiteCluster& cluster )
    {
      auto station = cluster.channelID().station() - 1;
      auto layer   = cluster.channelID().layer() - 1;
      double dxDy;
      double dzDy;
      double xAtYEq0;
      double zAtYEq0;
      double yBegin;
      double yEnd;
      //--- this method allow to set the values
      aSector->trajectory( cluster.channelID().strip(), cluster.interStripFraction(), dxDy, dzDy, xAtYEq0, zAtYEq0,
                           yBegin, yEnd );
      float cos   = aSector->cosAngle();
      float error = aSector->pitch() / std::sqrt( 12.0 );
      if ( dzDy != 0 ) exit( 1 );
      m_hits.addHit( std::forward_as_tuple( cluster, dxDy, xAtYEq0, zAtYEq0, yBegin, yEnd, cos, error ), station,
                     layer );
    }

    inline const HitContainer& hits() const { return m_hits; }

    inline HitRange hits( unsigned int station, unsigned int layer ) const { return m_hits.range( station, layer ); }

    inline void sortByXAtYEq0()
    {
      m_hits.sort( []( const UT::Hit& lhs, const UT::Hit& rhs ) {
        return std::make_tuple( lhs.xAtYEq0(), lhs.lhcbID() ) < std::make_tuple( rhs.xAtYEq0(), rhs.lhcbID() );
      } );
    }

    inline void setOffsets()
    {
      assert( m_hits.is_sorted( []( const UT::Hit& lhs, const UT::Hit& rhs ) {
              return std::make_tuple( lhs.xAtYEq0(), lhs.lhcbID() ) <
                     std::make_tuple( rhs.xAtYEq0(), rhs.lhcbID() );} ) &&
          "HitContainer<UT::Hit> need to be sorted before calling setOffsets()" );

      // Prepare the HitContainer
      m_hits.setOffsets();
    }

  protected:
  private:
    HitContainer m_hits;
  };
}
#endif // PRKERNEL_UTHITHANDLER_H
