#ifndef PRKERNEL_UTHIT_H
#define PRKERNEL_UTHIT_H 1

// Include files
#include "Event/STLiteCluster.h"
#include "LHCbID.h"
#include "STDet/DeSTSector.h"
#include "TfKernel/LineHit.h"

/** @class UTHit UTHit.h PrKernel/UTHit.h
 *  Minimal Implementation of Upstream tracker hit for pattern recognition
 *  @author Renato Quagliani, Christoph Hasse
 *  @date   2016-11-18
 */

namespace UT
{

  class Hit final
  {
  public:
    // constructor
    Hit( const LHCb::STLiteCluster& cluster, double dxDy, double xat0, double zat0, double yBegin,
         double yEnd, double cos, double error )
        : m_cos( cos )
        , m_dxDy( dxDy )
        , m_weight( 1. / error )
        , m_xAtYEq0( xat0 )
        , m_yBegin( yBegin )
        , m_yEnd( yEnd )
        , m_zAtYEq0( zat0 )
        , m_x( xAt( yMid() ) )
        , m_cluster( cluster )
    {
    }

    inline float cos() const { return m_cos; }
    inline float cosT() const { return ( fabs( m_xAtYEq0 ) < 1.0E-9 ) ? 1. / std::sqrt( 1 + m_dxDy * m_dxDy ) : cos(); }
    inline float dxDy() const { return m_dxDy; }
    inline bool highThreshold() const { return m_cluster.highThreshold(); }
    inline bool isYCompatible( const float y, const float tol ) const { return yMin() - tol <= y && y <= yMax() + tol; }
    inline bool isNotYCompatible( const float y, const float tol ) const { return yMin() - tol > y || y > yMax() + tol; }
    inline LHCb::LHCbID lhcbID() const { return LHCb::LHCbID( m_cluster.channelID() ); }
    inline int planeCode() const { return 2 * ( m_cluster.station() - 1 ) + ( m_cluster.layer() - 1 ) % 2; }
    inline float sinT() const { return tanT() * cosT(); }
    inline int size() const { return m_cluster.pseudoSize(); }
    inline float tanT() const { return -m_dxDy; }
    inline float weight() const { return m_weight * m_weight; }
    inline float xAt( const float globalY ) const { return m_xAtYEq0 + globalY * m_dxDy; }
    inline float xAtYEq0() const { return m_xAtYEq0; }
    inline float xAtYMid() const { return m_x; }
    inline float xMax() const { return std::max( xAt( yBegin() ), xAt( yEnd() ) ); }
    inline float xMin() const { return std::min( xAt( yBegin() ), xAt( yEnd() ) ); }
    inline float xT() const { return cos(); }
    inline float yBegin() const { return m_yBegin; }
    inline float yEnd() const { return m_yEnd; }
    inline float yMax() const { return std::max( yBegin(), yEnd() ); }
    inline float yMid() const { return 0.5 * ( yBegin() + yEnd() ); }
    inline float yMin() const { return std::min( yBegin(), yEnd() ); }
    inline float zAtYEq0() const { return m_zAtYEq0; }

  private:
    float m_cos;
    float m_dxDy;    ///< The dx/dy value
    float m_weight;  ///< The hit weight (1/error)
    float m_xAtYEq0; ///< The value of x at the point y=0
    float m_yBegin;  ///< The y value at the start point of the line
    float m_yEnd;    ///< The y value at the end point of the line
    float m_zAtYEq0; ///< The value of z at the point y=0
    float m_x;
    LHCb::STLiteCluster m_cluster;
  };

  typedef std::vector<const Hit*> Hits;

  namespace Mut
  {
    // Mutable UTHit to allow a modifiable object, needed in some algorithms.
    // Maybe it is also usefull to use a template and merge with the ModPrHit from the seeding
    struct Hit {

      const UT::Hit* HitPtr;
      float x, z;
      Tf::HitBase::StatusFlag status;
      float projection;

      Hit( const UT::Hit* ptr, float x, float z ) : HitPtr( ptr ), x( x ), z( z ) {}

      Hit( const UT::Hit* ptr, float x, float z, float proj, Tf::HitBase::EStatus stat )
          : HitPtr( ptr ), x( x ), z( z ), projection( proj )
      {
        status.set( stat, false );
      }
    };

    typedef std::vector<Hit> Hits;
    
    auto IncreaseByProj = []( const Hit& lhs, const Hit& rhs ) {
      if ( lhs.projection < rhs.projection ) return true;
      if ( rhs.projection < lhs.projection ) return false;
      return lhs.HitPtr->lhcbID() < rhs.HitPtr->lhcbID();
    };
  }
}

#endif // PRKERNEL_UTHIT_H
