// Include files

// local
#include "MagneticFieldGrid.h"

//-----------------------------------------------------------------------------
// Implementation file for class : MagneticFieldGrid
//
// 2008-07-26 : Marco Cattaneo
// 2009-06-05 : Wouter Hulsbergen
//-----------------------------------------------------------------------------

//=============================================================================
// Return the field vector fvec at the point xyz
//=============================================================================

namespace LHCb
{


  FieldVector
  MagneticFieldGrid::fieldVectorClosestPoint( const XYZPoint& r ) const
  {

    const Vec4f _r( r.x(), r.y(), r.z(), 0.0 );

    const Vec4ui ijk ( truncate_to_int( ( ( _r -  m_min_FL_V ) * m_invDxyz_V ) + 0.5 ) );
    const auto _i(ijk[0]), _j(ijk[1]), _k(ijk[2]);

    // get the field
    const Vec4f fvec_V =
      ( _i < m_Nxyz_V[0] && _j < m_Nxyz_V[1] &&  _k < m_Nxyz_V[2] ?
        m_scaleFactor_V * ( m_Q_V[m_Nxyz_V[0]*( m_Nxyz_V[1]*_k + _j ) + _i] ) :
        Vec4f(0.,0.,0.,0.) );

    return { fvec_V[0], fvec_V[1], fvec_V[2] };
  }

  FieldGradient
  MagneticFieldGrid::fieldGradient( const XYZPoint& r ) const
  {

    FieldGradient _rc ;

    const Vec4f _r( r.x(), r.y(), r.z(), 0.0 );

    const Vec4ui ijk ( truncate_to_int( ( _r - m_min_FL_V ) * m_invDxyz_V ) );
    const auto _i(ijk[0]), _j(ijk[1]), _k(ijk[2]);

    if ( _i < (m_Nxyz_V[0]-1) &&
         _j < (m_Nxyz_V[1]-1) &&
         _k < (m_Nxyz_V[2]-1) ) {
      const auto ijk000 = ( m_Nxyz_V[0]*( m_Nxyz_V[1]*_k + _j ) + _i );

      const Vec4i dijk( 1, m_Nxyz_V[0], m_Nxyz_V[0] * m_Nxyz_V[1], 0 );
      for ( int col = 0; col<3; ++col ) {
        const auto dQdX = ( m_Q_V[ ijk000 + dijk[col] ] - m_Q_V[ ijk000 ] ) * m_scaleFactor_V;
        _rc(0,col) = dQdX[0] * m_invDxyz_V[col] ;
        _rc(1,col) = dQdX[1] * m_invDxyz_V[col] ;
        _rc(2,col) = dQdX[2] * m_invDxyz_V[col] ;
      }
    }

    return _rc ;
  }



  void MagneticFieldGrid::fieldVector( const XYZPoint& r, XYZVector &   fvec ) const
  {
    FieldVector temp = fieldVector(r);
    fvec = temp;

  }


  FieldVector MagneticFieldGrid::fieldVector( const XYZPoint& r ) const
  {
    // //  Interpolate the field on a cube

    // Vectorised way

    Vec4f bf_V(0.,0.,0.,0.);

    const Vec4f _r( r.x(), r.y(), r.z(), 0.0 );

    const auto abc = ( _r - m_min_FL_V ) * m_invDxyz_V;
    const Vec4ui ijk ( truncate_to_int( abc ) );
    const auto _i(ijk[0]), _j(ijk[1]), _k(ijk[2]);

    if ( _i < m_Nxyz_V[0]-1 &&
         _j < m_Nxyz_V[1]-1 &&
         _k < m_Nxyz_V[2]-1 ) {

      // auxiliary variables defined at the vertices of the cube that
      // contains the (x, y, z) point where the field is interpolated
      const auto ijk000 = m_Nxyz_V[0]*( m_Nxyz_V[1]*_k     + _j )     + _i ;
      const auto ijk001 = m_Nxyz_V[0]*( m_Nxyz_V[1]*(_k+1) + _j )     + _i ;
      const auto ijk010 = m_Nxyz_V[0]*( m_Nxyz_V[1]*_k     + _j + 1 ) + _i ;
      const auto ijk011 = m_Nxyz_V[0]*( m_Nxyz_V[1]*(_k+1) + _j + 1 ) + _i ;

      const auto h1 = abc - Vec4f( _i, _j, _k, 0.0 );
      const auto h0 = 1.0f - h1;

      const auto h1x(h1[0]), h1y(h1[1]), h1z(h1[2]);
      const auto h0x(h0[0]), h0y(h0[1]), h0z(h0[2]);

      const auto h000 = h0x*h0y*h0z;
      const auto h001 = h0x*h0y*h1z;
      const auto h010 = h0x*h1y*h0z;
      const auto h011 = h0x*h1y*h1z;

      const auto h100 = h1x*h0y*h0z;
      const auto h101 = h1x*h0y*h1z;
      const auto h110 = h1x*h1y*h0z;
      const auto h111 = h1x*h1y*h1z;

      bf_V = ( ( h000*m_Q_V[ijk000] + h100*m_Q_V[ijk000+1] ) +
               ( h001*m_Q_V[ijk001] + h101*m_Q_V[ijk001+1] ) +
               ( h010*m_Q_V[ijk010] + h110*m_Q_V[ijk010+1] ) +
               ( h011*m_Q_V[ijk011] + h111*m_Q_V[ijk011+1] ) ) * m_scaleFactor_V;

      // Try a few alternatives to see which is faster...

      // const Vec4f a(h0y,h0y,h1y,h1y);
      // const Vec4f b(h0z,h1z,h0z,h1z);
      // const auto H = Vec8f(Vec4f(h0x),Vec4f(h1x)) * Vec8f(a,a) * Vec8f(b,b);

      // bf_V = ( ( H[0]*m_Q_V[ijk000] + H[4]*m_Q_V[ijk000+1] ) +
      //          ( H[1]*m_Q_V[ijk001] + H[5]*m_Q_V[ijk001+1] ) +
      //          ( H[2]*m_Q_V[ijk010] + H[6]*m_Q_V[ijk010+1] ) +
      //          ( H[3]*m_Q_V[ijk011] + H[7]*m_Q_V[ijk011+1] ) ) * m_scaleFactor_V;

      // const auto A = Vec4f(h0y,h0y,h1y,h1y) * Vec4f(h0z,h1z,h0z,h1z);
      // const auto H = h0x * A;
      // const auto I = h1x * A;

      // bf_V = ( ( H[0]*m_Q_V[ijk000] + I[0]*m_Q_V[ijk000+1] ) +
      //          ( H[1]*m_Q_V[ijk001] + I[1]*m_Q_V[ijk001+1] ) +
      //          ( H[2]*m_Q_V[ijk010] + I[2]*m_Q_V[ijk010+1] ) +
      //          ( H[3]*m_Q_V[ijk011] + I[3]*m_Q_V[ijk011+1] ) ) * m_scaleFactor_V;

    }

    return { bf_V[0], bf_V[1], bf_V[2] };
  }
}
