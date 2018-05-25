#ifndef TRACKEXTRAPOLATORS_TrackRungeKuttaExtrapolator_H
#define TRACKEXTRAPOLATORS_TrackRungeKuttaExtrapolator_H

// STL
#include <vector>
#include <array>
#include <iomanip>

#include "TrackFieldExtrapolatorBase.h"

// Eigen Objects
#include "LHCbMath/EigenTypes.h"

namespace RK {

namespace Scheme {
  enum class scheme_t { CashKarp, Fehlberg, DormandPrice, BogackiShampine, HeunEuler };
  // provide support for Gaudi::Property<scheme_t>
  std::string toString(const scheme_t& scheme);
  std::ostream& toStream(const scheme_t& scheme, std::ostream& os)
  { return os << std::quoted(toString(scheme),'\''); }
  StatusCode parse(scheme_t& result, const std::string& input );
  // and allow printout..
  inline std::ostream& operator<<(std::ostream& os, const scheme_t& s) { return toStream(s,os); }
}

using Scheme::scheme_t;

namespace details {

  /// The default floating point precision to use with the Eigen types
  using FloatType = double;

  /// Basically just a wrapper around the Eigen class, but Zero default constructed...
  template< typename TYPE, int ROWS, int COLUMNS >
  class Matrix : public ::Eigen::Matrix<TYPE,ROWS,COLUMNS>
  {
    typedef ::Eigen::Matrix<TYPE,ROWS,COLUMNS> Base;
  public:
    /// Default constructor adds zero initialisation
    Matrix() : Base( Base::Zero() ) { }
    /// forward to base constructor
    using Base::Base;
  };

  /// Type for a 4-vector
  template < typename TYPE = FloatType >
  using Vec4 = Matrix<TYPE,4,1>;

  /// Type for a 4 by 3 Matrix
  template < typename TYPE = FloatType >
  using Matrix43 = Matrix<TYPE,4,3>;

  /// Represenation of a State
  template < typename TYPE = FloatType >
  struct State final
  {
    State() = default;
    State( const Vec4<TYPE>& _vec, const FloatType _qop, const FloatType _z )
      : parameters(_vec), qop(_qop), z(_z) { }
    Vec4<TYPE> parameters;
    FloatType qop{0} ;
    FloatType z  {0} ;
    TYPE&  x()       noexcept { return parameters(0) ; }
    TYPE&  y()       noexcept { return parameters(1) ; }
    TYPE& tx()       noexcept { return parameters(2) ; }
    TYPE& ty()       noexcept { return parameters(3) ; }
    TYPE  tx() const noexcept { return parameters(2) ; }
    TYPE  ty() const noexcept { return parameters(3) ; }
  } ;

  template < typename TYPE = FloatType >
  struct Stage final
  {
    State<TYPE> state ;
    State<TYPE> derivative ;
    TrackFieldExtrapolatorBase::FieldVector Bfield ;
  } ;

  template < size_t numStages = 7, typename TYPE = FloatType >
  struct Cache final
  {
    std::array<Stage<TYPE>,numStages> stage;
    int laststep{-1} ;
    int step{0} ;
  } ;

  template < typename TYPE = FloatType >
  struct Jacobian final {

    Matrix43<TYPE> matrix;

    TYPE& dXdTx0()  noexcept { return matrix(0,0) ; }
    TYPE& dYdTx0()  noexcept { return matrix(1,0) ; }
    TYPE& dTxdTx0() noexcept { return matrix(2,0) ; }
    TYPE& dTydTx0() noexcept { return matrix(3,0) ; }

    TYPE& dXdTy0()  noexcept { return matrix(0,1) ; }
    TYPE& dYdTy0()  noexcept { return matrix(1,1) ; }
    TYPE& dTxdTy0() noexcept { return matrix(2,1) ; }
    TYPE& dTydTy0() noexcept { return matrix(3,1) ; }

    TYPE& dXdQoP0()  noexcept { return matrix(0,2) ; }
    TYPE& dYdQoP0()  noexcept { return matrix(1,2) ; }
    TYPE& dTxdQoP0() noexcept { return matrix(2,2) ; }
    TYPE& dTydQoP0() noexcept { return matrix(3,2) ; }

    TYPE dTxdTx0() const noexcept { return matrix(2,0) ; }
    TYPE dTydTx0() const noexcept { return matrix(3,0) ; }

    TYPE dTxdTy0() const noexcept { return matrix(2,1) ; }
    TYPE dTydTy0() const noexcept { return matrix(3,1) ; }

    TYPE dTxdQoP0() const noexcept { return matrix(2,2) ; }
    TYPE dTydQoP0() const noexcept { return matrix(3,2) ; }
  } ;

}
}

class TrackRungeKuttaExtrapolator: public TrackFieldExtrapolatorBase {

public:

  /// enums
  enum RKErrorCode { RKSuccess, RKOutOfTolerance, RKCurling, RKExceededMaxNumSteps } ;

  /// Constructor
  using TrackFieldExtrapolatorBase::TrackFieldExtrapolatorBase;

  /// initialize
  StatusCode initialize() override;

  /// initialize
  StatusCode finalize() override;

  using TrackFieldExtrapolatorBase::propagate;

  /// Propagate a state vector from zOld to zNew
  /// Transport matrix is calulated when transMat pointer is not NULL
  StatusCode propagate( Gaudi::TrackVector& stateVec,
                        double zOld,
                        double zNew,
                        Gaudi::TrackMatrix* transMat,
                        const LHCb::Tr::PID pid = LHCb::Tr::PID::Pion() ) const override;

  // public methods that are not in the interface. used for debugging with the extrapolator tester

private:


  struct RKStatistics final
  {
    RKStatistics& operator+=( const RKStatistics& rhs )
    {
      minstep = std::min(minstep,rhs.minstep) ;
      maxstep = std::max(maxstep,rhs.maxstep) ;
      err += rhs.err ;
      numstep += rhs.numstep ;
      numfailedstep += rhs.numfailedstep ;
      numincreasedstep += rhs.numincreasedstep ;
      sumstep += rhs.sumstep ;
      return *this ;
    }
    double sumstep{0} ;
    double minstep{1e9} ;
    double maxstep{0} ;
    size_t numstep{0} ;
    size_t numfailedstep{0} ;
    size_t numincreasedstep{0} ;
    RK::details::Vec4<> err;
  } ;
  const RKStatistics& statistics() const { return m_stats ; }

private:

  template <typename Tableaux>
  RKErrorCode extrapolate( const Tableaux& butcher, RK::details::State<>& state, double zout,
                           RK::details::Jacobian<>* jacobian, std::vector<double>* stepvector = nullptr ) const ;
  template <typename Tableaux>
  RKErrorCode extrapolateNumericalJacobian(const Tableaux& butcher,  RK::details::State<>& state, double zout, RK::details::Jacobian<>& jacobian) const ;

  template <typename Tableau>
  void evaluateRKStep(const Tableau& butcher,  double dz, RK::details::State<>& pin, RK::details::Vec4<>& err, RK::details::Cache<>& cache) const ;
  template <typename Tableau>
  void evaluateRKStepJacobian(const Tableau& butcher,  double dz, RK::details::Jacobian<>& jacobian,const RK::details::Cache<>& cache) const ;

private:

  // tool properties
  Gaudi::Property<double> m_toleranceX { this, "Tolerance", 0.001*Gaudi::Units::mm }; ///< required absolute position resolution
  Gaudi::Property<double> m_relToleranceTx { this, "RelToleranceTx", 5e-5 };           ///< required relative curvature resolution
  Gaudi::Property<double> m_minRKStep { this, "MinStep", 10*Gaudi::Units::mm };
  Gaudi::Property<double> m_maxRKStep { this, "MaxStep",  1*Gaudi::Units::m };
  Gaudi::Property<double> m_initialRKStep { this, "InitialStep", 1*Gaudi::Units::m };
  Gaudi::Property<double> m_sigma { this, "Sigma", 5.5 };
  Gaudi::Property<double> m_minStepScale { this, "MinStepScale", 0.125 };
  Gaudi::Property<double> m_maxStepScale { this, "MaxStepScale", 4.0  };
  Gaudi::Property<double> m_safetyFactor { this, "StepScaleSafetyFactor", 1.0 };
  Gaudi::Property<RK::scheme_t> m_rkscheme { this, "RKScheme", RK::scheme_t::CashKarp };
  Gaudi::Property<size_t> m_maxNumRKSteps { this, "MaxNumSteps" , 1000 };
  Gaudi::Property<bool>   m_correctNumSteps { this, "CorrectNumSteps", false };
  Gaudi::Property<bool>   m_numericalJacobian { this, "NumericalJacobian", false };
  Gaudi::Property<double> m_maxSlope { this, "MaxSlope", 10. };
  Gaudi::Property<double> m_maxCurvature { this, "MaxCurvature", 1/Gaudi::Units::m };

  // keep statistics for monitoring
  mutable unsigned long long m_numcalls{0} ;
  mutable RKStatistics m_totalstats ; // sum of stats for all calls
  mutable RKStatistics m_stats ;      // rkstats for the last call

};


#endif
