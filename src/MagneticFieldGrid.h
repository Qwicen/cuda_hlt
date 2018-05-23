#ifndef MAGFIELDGRID_H
#define MAGFIELDGRID_H

// STD
#include <vector>
#include <array>
#include <cmath>
#include "VectorClass/vectorclass.h"
#include "TROOT.h"
#include "TMath.h"
#include "TH1D.h"
#include "Math/Vector3D.h"
#include "Math/Point3D.h"
#include "Math/SMatrix.h"


    typedef ROOT::Math::XYZVector XYZVector;
    typedef ROOT::Math::SMatrix<double, 3, 3> Matrix3x3;
    typedef ROOT::Math::XYZPoint XYZPoint;
    typedef XYZVector FieldVector ;
    typedef Matrix3x3 FieldGradient ;



struct GridQuadrant
{
  std::vector<XYZVector> Q ;
  double zOffset {0};
  double Dxyz[3] {{}};
  size_t Nxyz[3] {{}};
} ;


class MagneticFieldGridReader ;

namespace LHCb
{

  /** @class MagneticFieldGrid MagneticFieldGrid.h
   *  Simple class to hold a magnetic field grid.
   *
   *  @author Marco Cattaneo, Wouter Hulsbergen
   *  @date   2008-07-26
   */
  class MagneticFieldGrid final
  {

  public:

    /// typedefs etc



    /// declare the field reader as friend so that we don't have to make a constructor
    friend class ::MagneticFieldGridReader;

    /// Return the field vector fvec at the point xyz by interpolation
    /// on the grid.
    FieldVector fieldVector( const XYZPoint& xyz ) const;

    //
    void fieldVector( const XYZPoint& xyz , XYZVector &   fvec ) const;

    /// Return the field vector fvec at the point xyz by interpolation
    /// on the grid.
    FieldGradient fieldGradient( const XYZPoint& xyz ) const;

    /// Return the field vector fvec at the point xyz by choosing the
    /// closest point on the grid.
    FieldVector fieldVectorClosestPoint( const XYZPoint& xyz ) const ;

    /// Return the magnetic field scale factor
    double scaleFactor() const noexcept { return m_scaleFactor ; }

    /// Update the scale factor
    void setScaleFactor( const double& s )
    {
      m_scaleFactor = s ;
      m_scaleFactor_V = Vec4f( s, s, s, 0. );
    }

private:
    template<class vectype, size_t... Is>
    std::array<vectype, 3> fetchVectorQ_helper (
      const vectype& indices,
      std::index_sequence<Is...>
    ) const {
      std::array<Vec4f, sizeof...(Is)> Qs {m_Q_V[indices[Is]]...};

      // TODO: This could be done more efficiently
      //       with templating magic and permute or similar
      return {
        vectype{Qs[Is][0]...},
        vectype{Qs[Is][1]...},
        vectype{Qs[Is][2]...}
      };
    }

    template<class vectype, size_t width>
    std::array<vectype, 3> fetchVectorQ (const vectype& indices) const {
      return fetchVectorQ_helper(indices, std::make_index_sequence<width>{});
    }

public:
    template<class vectype, size_t width>
    std::array<vectype, 3> horizontallyVectorizedFieldVectorClosestPoint (
      const std::array<vectype, 3>& point
    ) const {
      std::array<vectype, 3> bf {vectype{0.}, vectype{0.}, vectype{0.}};

      const std::array<vectype, 3> Nxyz {vectype{(double) m_Nxyz_V[0]}, vectype{(double) m_Nxyz_V[1]}, vectype{(double) m_Nxyz_V[2]}};
      const std::array<vectype, 3> inverted_Dxyz {vectype{m_invDxyz_V[0]}, vectype{m_invDxyz_V[1]}, vectype{m_invDxyz_V[2]}};
      const vectype x = (point[0] - vectype{m_min_FL_V[0]}) * inverted_Dxyz[0];
      const vectype y = (point[1] - vectype{m_min_FL_V[1]}) * inverted_Dxyz[1];
      const vectype z = (point[2] - vectype{m_min_FL_V[2]}) * inverted_Dxyz[2];
      const vectype i = floor(x);
      const vectype j = floor(y);
      const vectype k = floor(z);

      // if ( _i < m_Nxyz_V[0]-1 &&
      //      _j < m_Nxyz_V[1]-1 &&
      //      _k < m_Nxyz_V[2]-1 ) {
      const auto condition {i < Nxyz[0] - 1. &&
                            j < Nxyz[1] - 1. &&
                            k < Nxyz[2] - 1.};

      if (horizontal_or(condition)) {
        const vectype cond_multiplier = select(condition, vectype{1.}, vectype{0.});
        const vectype scaleFactor {m_scaleFactor};
        
        const vectype index = cond_multiplier * (Nxyz[0] * (Nxyz[1] * k + j) + i);
        const std::array<vectype, 3> q = fetchVectorQ<vectype, width>(index);


        bf[0] = cond_multiplier * scaleFactor * q[0];
        bf[1] = cond_multiplier * scaleFactor * q[1];
        bf[2] = cond_multiplier * scaleFactor * q[2];
      }

      return bf;
    }

    template<class vectype, size_t width>
    std::array<vectype, 3> horizontallyVectorizedFieldVector (
      const std::array<vectype, 3>& point
    ) const {
      std::array<vectype, 3> bf {vectype{0.}, vectype{0.}, vectype{0.}};

      // Load variables
      // TODO: This could be done more efficiently with
      //       templating magic and permute or so
      const std::array<vectype, 3> inverted_Dxyz {vectype{m_invDxyz_V[0]}, vectype{m_invDxyz_V[1]}, vectype{m_invDxyz_V[2]}};
      const std::array<vectype, 3> Nxyz {vectype{(double) m_Nxyz_V[0]}, vectype{(double) m_Nxyz_V[1]}, vectype{(double) m_Nxyz_V[2]}};
      const vectype scaleFactor {m_scaleFactor};

      // double x = r.x() - m_min_FL[0] ; unsigned i =  int(x/m_Dxyz[0]) ;
      // double y = r.y() - m_min_FL[1] ; unsigned j =  int(y/m_Dxyz[1]) ;
      // double z = r.z() - m_min_FL[2] ; unsigned k =  int(z/m_Dxyz[2]) ;
      const vectype x = (point[0] - vectype{m_min_FL_V[0]}) * inverted_Dxyz[0];
      const vectype y = (point[1] - vectype{m_min_FL_V[1]}) * inverted_Dxyz[1];
      const vectype z = (point[2] - vectype{m_min_FL_V[2]}) * inverted_Dxyz[2];

      const vectype i = floor(x);
      const vectype j = floor(y);
      const vectype k = floor(z);

      // if ( _i < m_Nxyz_V[0]-1 &&
      //      _j < m_Nxyz_V[1]-1 &&
      //      _k < m_Nxyz_V[2]-1 ) {
      const auto condition {i < Nxyz[0] - 1. &&
                            j < Nxyz[1] - 1. &&
                            k < Nxyz[2] - 1.};

      if (horizontal_or(condition)) {
        std::array<vectype, 3> q1;
        std::array<vectype, 3> q2;

        // Use a conditional multiplier to avoid more selects
        const vectype cond_multiplier = select(condition, vectype{1.}, vectype{0.});

        const vectype ijk000 = cond_multiplier * (Nxyz[0] * (Nxyz[1]*k     + j)     + i);
        const vectype ijk001 = cond_multiplier * (Nxyz[0] * (Nxyz[1]*(k+1) + j)     + i);
        const vectype ijk010 = cond_multiplier * (Nxyz[0] * (Nxyz[1]*k     + j + 1) + i);
        const vectype ijk011 = cond_multiplier * (Nxyz[0] * (Nxyz[1]*(k+1) + j + 1) + i);
        
        // float hx1 = ( x-i*m_Dxyz[0] )/m_Dxyz[0]; auto hx0 = 1.0f-hx1;
        // float hy1 = ( y-j*m_Dxyz[1] )/m_Dxyz[1]; auto hy0 = 1.0f-hy1;
        // float hz1 = ( z-k*m_Dxyz[2] )/m_Dxyz[2]; auto hz0 = 1.0f-hz1;
        const vectype hx1 = (x - i);
        const vectype hy1 = (y - j);
        const vectype hz1 = (z - k);
        
        const vectype hx0 = vectype{1.} - hx1;
        const vectype hy0 = vectype{1.} - hy1;
        const vectype hz0 = vectype{1.} - hz1;

        // auto h000 = hx0*hy0*hz0; auto h100 = hx1*hy0*hz0;
        // auto ijk000 = ( m_Nxyz[0]*( m_Nxyz[1]*k     + j )     + i );
        // bf  = h000*m_Q[ijk000]+h100*m_Q[ijk000+1] ;
        const vectype h000 = hx0 * hy0 * hz0;
        const vectype h100 = hx1 * hy0 * hz0;
        q1 = fetchVectorQ<vectype, width>(ijk000);
        q2 = fetchVectorQ<vectype, width>(ijk000 + 1);
        bf[0] = h000 * q1[0] + h100 * q2[0];
        bf[1] = h000 * q1[1] + h100 * q2[1];
        bf[2] = h000 * q1[2] + h100 * q2[2];

        // auto h001 = hx0*hy0*hz1; auto h101 = hx1*hy0*hz1;
        // auto ijk001 = ( m_Nxyz[0]*( m_Nxyz[1]*(k+1) + j )     + i );
        // bf += h001*m_Q[ijk001]+h101*m_Q[ijk001+1] ;
        const vectype h001 = hx0 * hy0 * hz1;
        const vectype h101 = hx1 * hy0 * hz1;
        q1 = fetchVectorQ<vectype, width>(ijk001);
        q2 = fetchVectorQ<vectype, width>(ijk001 + 1);
        bf[0] += h001 * q1[0] + h101 * q2[0];
        bf[1] += h001 * q1[1] + h101 * q2[1];
        bf[2] += h001 * q1[2] + h101 * q2[2];

        // auto h010 = hx0*hy1*hz0; auto h110 = hx1*hy1*hz0;
        // auto ijk010 = ( m_Nxyz[0]*( m_Nxyz[1]*k     + j + 1 ) + i );
        // bf += h010*m_Q[ijk010]+h110*m_Q[ijk010+1] ;
        const vectype h010 = hx0 * hy1 * hz0;
        const vectype h110 = hx1 * hy1 * hz0;
        q1 = fetchVectorQ<vectype, width>(ijk010);
        q2 = fetchVectorQ<vectype, width>(ijk010 + 1);
        bf[0] += h010 * q1[0] + h110 * q2[0];
        bf[1] += h010 * q1[1] + h110 * q2[1];
        bf[2] += h010 * q1[2] + h110 * q2[2];

        // auto h011 = hx0*hy1*hz1; auto h111 = hx1*hy1*hz1;
        // auto ijk011 = ( m_Nxyz[0]*( m_Nxyz[1]*(k+1) + j + 1)  + i );
        // bf += h011*m_Q[ijk011]+h111*m_Q[ijk011+1] ;
        const vectype h011 = hx0 * hy1 * hz1;
        const vectype h111 = hx1 * hy1 * hz1;
        q1 = fetchVectorQ<vectype, width>(ijk011);
        q2 = fetchVectorQ<vectype, width>(ijk011 + 1);
        bf[0] += h011 * q1[0] + h111 * q2[0];
        bf[1] += h011 * q1[1] + h111 * q2[1];
        bf[2] += h011 * q1[2] + h111 * q2[2];

        bf[0] *= scaleFactor * cond_multiplier;
        bf[1] *= scaleFactor * cond_multiplier;
        bf[2] *= scaleFactor * cond_multiplier;
      }

      return bf;
    }

  private:

    double m_scaleFactor = 1; ///< The scale factor

    Vec4f m_scaleFactor_V = { 1,1,1,0 };       ///< Vector version of the scale factor
    std::vector<Vec4f> m_Q_V;                  ///< Vectorised Field map
    Vec4f   m_min_FL_V = { 0.,0.,0.,0. };      ///< Offset in x, y and z
    Vec4f   m_Dxyz_V = { 0.,0.,0.,0. };        ///< Steps in x, y and z
    Vec4f   m_invDxyz_V = { 0.,0.,0.,0. };     ///< Inverse of steps in x, y and z (cached for speed)
    std::array<unsigned,3> m_Nxyz_V = {0,0,0}; ///< Number of steps in x, y and z

  };

}

// Instantiate horizontally vectorized entrypoints
#if defined(__SSE__)
template std::array<Vec2d, 3> LHCb::MagneticFieldGrid::horizontallyVectorizedFieldVector<Vec2d, 2>(
  const std::array<Vec2d, 3>&) const;
#endif
#if defined(__AVX__)
template std::array<Vec4d, 3> LHCb::MagneticFieldGrid::horizontallyVectorizedFieldVector<Vec4d, 4>(
  const std::array<Vec4d, 3>&) const;
#endif
#if defined(__AVX512F__)
template std::array<Vec8d, 3> LHCb::MagneticFieldGrid::horizontallyVectorizedFieldVector<Vec8d, 8>(
  const std::array<Vec8d, 3>&) const;
#endif

#endif // MAGFIELDGRID_H
