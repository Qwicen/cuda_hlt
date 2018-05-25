#include "TrackRungeKuttaExtrapolator.h"
#include "LHCbMath/FastRoots.h"

#include "GaudiKernel/PhysicalConstants.h"
#include "Kernel/STLExtensions.h"
#include "boost/optional.hpp"
#include "boost/variant.hpp"
#include <numeric>
#include <stdexcept>

//
//       (   1   0   dXdTx0   dXdTy0     dXdQoP0   )
//       (   0   1   dYdTx0   dYdTy0     dYdQoP0   )
//   J = (   0   0   dTxdTx0  dTxdTy0    dTxdQoP0  )
//       (   0   0   dTydTx0  dTxdTy0    dTydQoP0  )
//       (   0   0      0        0         1       )
//


namespace RK {
namespace { // make sure table has internal linkage only...
  static const auto table = LHCb::make_array(
          std::make_pair(scheme_t::CashKarp,         "CashKarp"),
          std::make_pair(scheme_t::Fehlberg,         "Fehlberg"),
          std::make_pair(scheme_t::DormandPrice,     "DormandPrice"),
          std::make_pair(scheme_t::BogackiShampine,  "BogackiShampine"),
          std::make_pair(scheme_t::HeunEuler,        "HeunEuler") );
}
namespace Scheme {
std::string toString(const scheme_t& scheme) {
    auto i = std::find_if( begin(table), end(table),
                           [&](const std::pair<scheme_t,const char*>& p)
                           { return p.first == scheme; } );
    if (i==end(table)) { throw std::range_error( "Invalid RK::scheme_t" ); return "<<<INVALID>>>"; }
    return i->second;
}

StatusCode parse(scheme_t& result, const std::string& input ) {
    auto i = std::find_if( begin(table), end(table),
                           [&](const std::pair<scheme_t,const char*>& p)
                           { return p.second == input; } );
    if (i==end(table)) return StatusCode::FAILURE;
    result = i->first;
    return StatusCode::SUCCESS;
}
}
}
namespace {


// *********************************************************************************************************
// Butcher tables for various adaptive Runge Kutta methods. These are all taken from wikipedia.
// *********************************************************************************************************

template <unsigned N, bool fsl, typename TYPE=double> struct RKButcherTableau {
  std::array<TYPE,N*(N-1)/2> a;
  std::array<TYPE,N>         b5;
  std::array<TYPE,N>         b4;
  static constexpr int numStages() { return N; }
  static constexpr bool firstSameAsLast() { return fsl; }
};

constexpr static const auto CashKarp = RKButcherTableau<6,false>{
  {1.0/5.0,
   3.0/40.0,       9.0/40.0,
   3.0/10.0,      -9.0/10.0, 6.0/5.0,
   -11.0/54.0,     5.0/2.0, -70.0/27.0, 35.0/27.0,
   1631.0/55296.0, 175.0/512.0, 575.0/13824.0,44275.0/110592.0, 253.0/4096.0},
  {37.0/378.0    , 0., 250.0/621.0,     125.0/594.0,     0.,            512.0/1771.0},
  {2825.0/27648.0, 0., 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0}
};
constexpr static const auto Fehlberg = RKButcherTableau<6,false>{
  {1/4.,
   3/32.,         9/32.,
   1932/2197.,         -7200/2197.,         7296/2197.,
   439/216.,         -8.,             3680/513.,         -845/4104.,
   -8/27.,         2.,                 -3544/2565.,         1859/4104.,         -11/40. },
  {25/216.,         0.,         1408/2565.,         2197/4104.,         -1/5.,            0. },
  {16/135.,         0.,         6656/12825.,         28561/56430.,         -9/50.,    2/55. }
};
constexpr static const auto DormandPrice = RKButcherTableau<7,true>{
  {1/5.,
   3/40.,          9/40.,
   44/45.,        -56/15.,      32/9.,
   19372/6561.,   -25360/2187., 64448/6561.,  -212/729.,
   9017/3168.,    -355/33.,     46732/5247.,  49/176.,    -5103/18656.,
   35/384.,        0.,          500/1113.,    125/192.,   -2187/6784.,   11/84. },
// I think wikipedia swapped them here, but it doesn't seem to make much difference. I don't understand that.
  {5179/57600.,    0.,          7571/16695.,  393/640.,   -92097/339200., 187/2100.,  1/40.},
  {35/384.,        0.,          500/1113.,    125/192.,   -2187/6784.,    11/84.,     0.}
};
constexpr static const auto BogackiShampine = RKButcherTableau<4,false>{
  {1/2.,
   0.,   3/4.,
   2/9., 1/3., 4/9. },
  {7/24., 1/4.,         1/3.,         1/8.},
  {2/9.,  1/3.,         4/9.,         0 }
};
constexpr static const auto HeunEuler = RKButcherTableau<2,false>{
  {1.},
  {1.0, 0. },
  {0.5, 0.5}
};

boost::variant< std::reference_wrapper<const RKButcherTableau<6,false>>,
                std::reference_wrapper<const RKButcherTableau<7,true>>,
                std::reference_wrapper<const RKButcherTableau<4,false>>,
                std::reference_wrapper<const RKButcherTableau<2,false>> >
select_tableau( RK::scheme_t s) {
  switch ( s ) {
    case RK::scheme_t::CashKarp:        return CashKarp;
    case RK::scheme_t::Fehlberg:        return Fehlberg;
    case RK::scheme_t::DormandPrice:    return DormandPrice;
    case RK::scheme_t::BogackiShampine: return BogackiShampine;
    case RK::scheme_t::HeunEuler:       return HeunEuler;
    default : throw("invalid RK::scheme_t");      return CashKarp; // should not be reached!
  };
}

RK::details::State<>
evaluateDerivatives(const RK::details::State<>& state,
                    const TrackFieldExtrapolatorBase::FieldVector& field )
{
  const auto tx  = state.tx() ;
  const auto ty  = state.ty() ;
  const auto qop = state.qop ;

  const auto Bx  = field.x() ;
  const auto By  = field.y() ;
  const auto Bz  = field.z() ;

  const auto tx2 = tx*tx;
  const auto ty2 = ty*ty;

  const auto norm = std::sqrt( 1 + tx2 + ty2 ) ;
  const auto Ax = norm * (  ty * ( tx*Bx + Bz ) - ( 1 + tx2) * By ) ;
  const auto Ay = norm * ( -tx * ( ty*By + Bz ) + ( 1 + ty2) * Bx ) ;

  // this is 'dState/Dz'
  //          x   y      tx      ty   qop  z
  return { { tx, ty, qop*Ax, qop*Ay }, 0, 1 };
}

RK::details::Jacobian<>
evaluateDerivativesJacobian( const RK::details::State<>& state,
                             const RK::details::Jacobian<>& jacobian,
                             const TrackFieldExtrapolatorBase::FieldVector& field )
{
  const auto tx  = state.tx() ;
  const auto ty  = state.ty() ;
  const auto qop = state.qop ;

  const auto Bx  = field.x() ;
  const auto By  = field.y() ;
  const auto Bz  = field.z() ;

  const auto tx2 = tx*tx ;
  const auto ty2 = ty*ty ;

  const auto n2  = 1 + tx2 + ty2 ;
  const auto n   = std::sqrt( n2 ) ;

  const auto txBx = tx*Bx;
  const auto txBy = tx*By;
  const auto tyBy = ty*By;
  const auto tyBx = ty*Bx;

  const auto Ax  = n * (  ty * ( txBx + Bz ) - ( 1 + tx2) * By ) ;
  const auto Ay  = n * ( -tx * ( tyBy + Bz ) + ( 1 + ty2) * Bx ) ;

  const auto Ax_n2 = Ax/n2;
  const auto Ay_n2 = Ay/n2;

  // now we compute 'dJacobian/dZ'
  const auto dAxdTx = Ax_n2*tx + n * ( tyBx - 2*txBy ) ;
  const auto dAxdTy = Ax_n2*ty + n * ( txBx  + Bz ) ;

  const auto dAydTx = Ay_n2*tx + n * ( -tyBy - Bz ) ;
  const auto dAydTy = Ay_n2*ty + n * ( -txBy + 2*tyBx) ;

  // we'll do the factors of c later
  RK::details::Jacobian<> jacobianderiv;

  // derivatives to Tx0
  jacobianderiv.dXdTx0()  = jacobian.dTxdTx0() ;
  jacobianderiv.dYdTx0()  = jacobian.dTydTx0() ;
  jacobianderiv.dTxdTx0() = qop * ( jacobian.dTxdTx0() * dAxdTx +
                                    jacobian.dTydTx0() * dAxdTy ) ;
  jacobianderiv.dTydTx0() = qop * ( jacobian.dTxdTx0() * dAydTx +
                                    jacobian.dTydTx0() * dAydTy ) ;

  // derivatives to Ty0
  jacobianderiv.dXdTy0()  = jacobian.dTxdTy0() ;
  jacobianderiv.dYdTy0()  = jacobian.dTydTy0() ;
  jacobianderiv.dTxdTy0() = qop * ( jacobian.dTxdTy0() * dAxdTx +
                                    jacobian.dTydTy0() * dAxdTy ) ;
  jacobianderiv.dTydTy0() = qop * ( jacobian.dTxdTy0() * dAydTx +
                                    jacobian.dTydTy0() * dAydTy ) ;

  // derivatives to qopc
  jacobianderiv.dXdQoP0()  = jacobian.dTxdQoP0() ;
  jacobianderiv.dYdQoP0()  = jacobian.dTydQoP0() ;
  jacobianderiv.dTxdQoP0() = Ax + qop * ( jacobian.dTxdQoP0() * dAxdTx +
                                          jacobian.dTydQoP0() * dAxdTy ) ;
  jacobianderiv.dTydQoP0() = Ay + qop * ( jacobian.dTxdQoP0() * dAydTx +
                                          jacobian.dTydQoP0() * dAydTy ) ;
  return jacobianderiv;
}
}

// *********************************************************************************************************

DECLARE_COMPONENT( TrackRungeKuttaExtrapolator )

StatusCode
TrackRungeKuttaExtrapolator::finalize()
{
  if( UNLIKELY( msgLevel(MSG::DEBUG) ) ) {
    debug() << "Number of calls:     " << m_numcalls << endmsg ;
    debug() << "Min step length:     " << m_totalstats.minstep << endmsg ;
    debug() << "Max step length:     " << m_totalstats.maxstep << endmsg ;
    debug() << "Av step length:      " << m_totalstats.sumstep/(m_totalstats.numstep-m_totalstats.numfailedstep) << endmsg ;
    debug() << "Av num step:         " << m_totalstats.numstep/double(m_numcalls) << endmsg ;
    debug() << "Fr. failed steps:    " << m_totalstats.numfailedstep/double(m_totalstats.numstep) << endmsg ;
    debug() << "Fr. increased steps: " << m_totalstats.numincreasedstep/double(m_totalstats.numstep) << endmsg ;
  }
  return TrackFieldExtrapolatorBase::finalize() ;
}

StatusCode
TrackRungeKuttaExtrapolator::initialize()
{
  StatusCode sc = TrackFieldExtrapolatorBase::initialize();
  if( UNLIKELY( msgLevel(MSG::DEBUG) ) )
    debug() << "Using RK scheme: " << m_rkscheme.value() << endmsg ;

  // reset counters
  m_totalstats = RKStatistics() ;
  m_numcalls = 0 ;

  return sc ;
}

// Propagate a state vector from zOld to zNew
// Transport matrix is calulated when transMat pointer is not NULL
StatusCode
TrackRungeKuttaExtrapolator::propagate( Gaudi::TrackVector& state,
                                        double zin,
                                        double zout,
                                        Gaudi::TrackMatrix* transMat,
                                        const LHCb::Tr::PID /*pid*/ ) const
{
  // Bail out if already at destination
  if ( std::abs(zin-zout) < TrackParameters::propagationTolerance ) {
    if( transMat ) *transMat = ROOT::Math::SMatrixIdentity();
    return StatusCode::SUCCESS ;
  }

  boost::optional< RK::details::Jacobian<> > jacobian;
  if (transMat) jacobian.emplace();

  // translate the state to one we use in the runge kutta. note the factor c.
  RK::details::State<> rkstate( { state(0),state(1),state(2),state(3) },
                                state(4) * Gaudi::Units::c_light, zin );

  auto tableau = select_tableau( m_rkscheme.value() );
  RKErrorCode success = ( m_numericalJacobian && jacobian
                            ? extrapolateNumericalJacobian(tableau, rkstate, zout, *jacobian)
                            : extrapolate(tableau, rkstate, zout, jacobian.get_ptr() ) );
  if ( success != RKSuccess ) {
    return Warning("RungeKuttaExtrapolator failed with code: "
                 + std::to_string( success  ),
                 StatusCode::FAILURE,0) ;
  }
  // translate the state back
  //info() << "In  " << state(0) << " " << state(1) << " " << state(2) << " " << state(3) << endmsg;
  state(0) = rkstate.x() ;
  state(1) = rkstate.y() ;
  state(2) = rkstate.tx() ;
  state(3) = rkstate.ty() ;
  //info() << "Out " << state(0) << " " << state(1) << " " << state(2) << " " << state(3) << endmsg;

  if ( transMat ) {
    *transMat = Gaudi::TrackMatrix() ;
    (*transMat)(0,0) = 1 ;
    (*transMat)(1,1) = 1 ;
    (*transMat)(4,4) = 1 ;
    for ( int irow=0; irow<4; ++irow ) {
      for ( int icol=0; icol<3; ++icol ) {
        (*transMat)(irow,icol+2) = jacobian->matrix(irow,icol) ;
      }
      (*transMat)(irow,4) *= Gaudi::Units::c_light ;
    }
  }
  return StatusCode::SUCCESS ;
}


template <typename Tableaux>
TrackRungeKuttaExtrapolator::RKErrorCode
TrackRungeKuttaExtrapolator::extrapolate( const Tableaux& table,
                                          RK::details::State<>& state,
                                          double zout,
                                          RK::details::Jacobian<>* jacobian,
                                          std::vector<double>* stepvector) const
{
  // count calls
  ++m_numcalls ;

  // initialize the jacobian
  if ( jacobian ) {
    jacobian->dTxdTx0() = 1 ;
    jacobian->dTydTy0() = 1 ;
  }

  // now start stepping. first try with a single step. this may not be
  // very optimal inside the magnet.
  const auto totalStep = zout - state.z ;
  //auto toleranceTx = std::abs(m_toleranceX/totalStep) ;
  auto toleranceX  = m_toleranceX.value() ;
  auto toleranceTx = toleranceX/std::abs(totalStep) ;

  auto absstep = std::min( std::abs(totalStep), m_initialRKStep.value() ) ;
  const auto direction = totalStep > 0 ? +1 : -1 ;
  bool laststep = absstep < m_minRKStep ;

  RK::details::Cache<> rkcache ;
  RK::details::Vec4<> err, totalErr;
  RKStatistics  stats ;
  RKErrorCode rc = RKSuccess ;

  while ( rc==RKSuccess && std::abs(state.z - zout) > TrackParameters::propagationTolerance ) {

    // make a single range-kutta step
    auto prevstate = state ;
    boost::apply_visitor( [&](const auto& tbl) {
          this->evaluateRKStep(tbl.get(), absstep * direction, state, err, rkcache);
    }, table );

    // decide if the error is small enough

    // always accept the step if it is smaller than the minimum step size
    bool success = (absstep <= m_minRKStep) ;
    if( !success ) {
      if ( m_correctNumSteps ) {
        const auto estimatedN = std::abs(totalStep) / absstep ;
        toleranceX  = (m_toleranceX/estimatedN/m_sigma) ;
        toleranceTx = toleranceX/std::abs(totalStep) ;
        //(m_toleranceX/10000)/estimatedN/m_sigma ;
      }

      // apply the acceptance criterion.
      auto normdx  = std::abs( err(0) ) / toleranceX ;
      auto normdy  = std::abs( err(1) ) / toleranceX ;
      auto deltatx = state.tx() - prevstate.tx() ;
      auto normdtx = std::abs( err(2) ) / ( toleranceTx + std::abs( deltatx ) * m_relToleranceTx ) ;
      auto errorOverTolerance = std::max( normdx, std::max( normdy, normdtx ) ) ;
      success = (errorOverTolerance <= m_sigma) ;
      //     std::cout << "step: " << rkcache.step << " " << success << " "
      //                 << prevstate.z << " "
      //                 << state.z << " " << absstep << " "
      //                 << errorOverTolerance << std::endl ;

      // do some stepping monitoring, before adapting step size
      if(success) {
        stats.sumstep += absstep ;
        if(!laststep) stats.minstep = std::min( stats.minstep, absstep ) ;
        stats.maxstep = std::max( stats.maxstep, absstep ) ;
      } else {
        ++stats.numfailedstep ;
      }

      // adapt the stepsize if necessary. the powers come from num.recipees.
      double stepfactor(1) ;
      if( errorOverTolerance > 1 ) { // decrease step size
        stepfactor = std::max( m_minStepScale.value(), m_safetyFactor / std::sqrt(std::sqrt(errorOverTolerance))); // was : * std::pow( errorOverTolerance , -0.25 ) ) ;
      } else {                       // increase step size
        if( errorOverTolerance > 0 ) {
          stepfactor = std::min( m_maxStepScale.value(), m_safetyFactor * FastRoots::invfifthroot(errorOverTolerance) ) ; // was: * std::pow( errorOverTolerance, -0.2) ) ;
        } else {
          stepfactor = m_maxStepScale ;
        }
        ++stats.numincreasedstep ;
      }

      // apply another limitation criterion
      absstep = std::max(m_minRKStep.value(),
                std::min(absstep*stepfactor,
                         m_maxRKStep.value()));
    }

    //info() << "Success = " << success << endmsg;
    if ( success ) {
      // if we need the jacobian, evaluate it only for successful steps
      auto thisstep = state.z - prevstate.z ; // absstep has already been changed!
      if ( jacobian ) {
          boost::apply_visitor( [&](const auto& tbl) {
            this->evaluateRKStepJacobian( tbl.get(), thisstep, *jacobian, rkcache ) ;
          } , table );
      }
      // update the step, to invalidate the cache (or reuse the last stage)
      ++rkcache.step;
      if (stepvector) stepvector->push_back( thisstep ) ;
      stats.err += err ;
    } else {
      // if this step failed, don't update the state
      state = prevstate ;
    }

    // check that we don't step beyond the target
    if( absstep > direction * (zout - state.z) ) {
      absstep = std::abs(zout - state.z) ;
      laststep = true ;
    }

    // final check: bail out for vertical or looping tracks
    if( std::max(std::abs(state.tx()), std::abs(state.ty())) > m_maxSlope ) {
      if( UNLIKELY( msgLevel(MSG::DEBUG) ) )
        debug() << "State has very large slope, probably curling: tx, ty = "
                << state.tx() << ", " << state.ty()
                << " z_origin, target, current: "
                << zout - totalStep << " " << zout << " " << state.z
                << endmsg ;
      rc = RKCurling ;
    } else if( std::abs(state.qop * rkcache.stage[0].Bfield.y() ) > m_maxCurvature ) {
      if( UNLIKELY( msgLevel(MSG::DEBUG) ) )
        debug() << "State has too small curvature radius: "
                << state.qop * rkcache.stage[0].Bfield.y()
                << " z_origin, target, current: "
                << zout - totalStep << " " << zout << " " << state.z
                << endmsg ;
      rc = RKCurling ;
    } else if( stats.numfailedstep + rkcache.step  >= m_maxNumRKSteps ) {
      if( UNLIKELY( msgLevel(MSG::DEBUG) ) )
        debug() << "Exceeded max numsteps. " << endmsg ;
      rc = RKExceededMaxNumSteps ;
    }
  }

  stats.numstep = rkcache.step ;
  m_stats = stats ;
  m_totalstats += stats ;

  return rc ;
}

template <typename Tableau>
void
TrackRungeKuttaExtrapolator::evaluateRKStep( const Tableau& table,
                                             double dz,
                                             RK::details::State<>& pin,
                                             RK::details::Vec4<>& err,
                                             RK::details::Cache<>& cache) const
{
  //  debug() << "z-component of input: "
  //          << pin.z << " " << dz << endmsg ;

  std::array< RK::details::Vec4<>, Tableau::numStages() > k;
  int firststage(0) ;

  // previous step failed, reuse the first stage
  if( cache.laststep == cache.step ) {
    firststage = 1 ;
    k[0] = dz * cache.stage[0].derivative.parameters ;
    //assert( std::abs(pin.z - cache.stage[0].state.z) < 1e-4 ) ;
  }
  // previous step succeeded and we can reuse the last stage (Dormand-Price)
  else if ( Tableau::firstSameAsLast() && cache.step > 0 ) {
    firststage = 1 ;
    cache.stage[0] = cache.stage[Tableau::numStages()-1] ;
    k[0] = dz * cache.stage[0].derivative.parameters ;
  }
  cache.laststep = cache.step ;

  for( int m = firststage ; m != Tableau::numStages() ; ++m ) {
    auto& stage = cache.stage[m];
    // evaluate the state
    stage.state = std::inner_product( begin(k), std::next(begin(k),m),
                                      std::next(begin(table.a),m*(m-1)/2),
                                      pin,
                                      [](RK::details::State<> lhs, const RK::details::State<>& rhs) {
                                          lhs.parameters += rhs.parameters;
                                          lhs.z          += rhs.z;
                                          return lhs;
                                      },
                                      [dz](const RK::details::Vec4<>& rhs, double a ) -> RK::details::State<> {
                                          return { a*rhs, 0, a*dz };
                                      } );
    // evaluate the derivatives
    //std::cout << "stage " << m << " --> " << stage.state.z << std::endl ;
    if ( UNLIKELY( std::abs( stage.state.x() ) > 1e6 ||
                   std::abs( stage.state.y() ) > 1e6 ||
                   std::abs( stage.state.z   ) > 1e6 ))
    {
      Info( "Very large value (> 1e6 mm) for state position. Breaking iteration", StatusCode::FAILURE,1).ignore();
      break;
    }
    stage.Bfield = fieldVector( Gaudi::XYZPoint(stage.state.x(),
                                                stage.state.y(),
                                                stage.state.z ) ) ;
    stage.derivative = evaluateDerivatives( stage.state, stage.Bfield ) ;
    k[m] = dz * stage.derivative.parameters ;
  }

  // update state and error
  err.fill(0);
  for( int m = 0 ; m!=Tableau::numStages(); ++m ) {
    // this is the difference between the 4th and 5th order
    err  += (table.b5[m] - table.b4[m] ) * k[m] ;
    // this is the fifth order change
    pin.parameters += table.b5[m] * k[m] ;
  }

  pin.z += dz ;
}

template <typename Tableau>
void
TrackRungeKuttaExtrapolator::evaluateRKStepJacobian( const Tableau& table,
                                                     double dz,
                                                     RK::details::Jacobian<>& jacobian,
                                                     const RK::details::Cache<>& cache) const
{
  // evaluate the jacobian. not that we never resue last stage
  // here. that's not entirely consistent (but who cares)
  std::array< RK::details::Matrix43<>, Tableau::numStages() > k;
  for ( int m = 0; m!=Tableau::numStages(); ++m ) {
    // evaluate the derivatives. reuse the parameters and bfield from the cache
    k[m] = dz * evaluateDerivativesJacobian( cache.stage[m].state,
                                             std::inner_product( begin(k), std::next(begin(k),m),
                                                                 std::next( begin(table.a), m*(m-1)/2 ),
                                                                 jacobian,
                                                                 [](RK::details::Jacobian<> j, const RK::details::Matrix43<>& wk) {
                                                                     j.matrix += wk;
                                                                     return j;
                                                                 },
                                                                 [](const RK::details::Matrix43<>& ki, double a) {
                                                                     return a*ki;
                                                                 }),
                                             cache.stage[m].Bfield ).matrix;
  }

  for( int m = 0 ; m!=Tableau::numStages(); ++m ) {
    jacobian.matrix += table.b5[m] * k[m] ;
  }
}


template <typename Tableau>
TrackRungeKuttaExtrapolator::RKErrorCode
TrackRungeKuttaExtrapolator::extrapolateNumericalJacobian( const Tableau& table,
                                                           RK::details::State<>& state,
                                                           double zout,
                                                           RK::details::Jacobian<>& jacobian) const
{
  // call the stanndard method but store the steps taken
  size_t cachednumstep(m_stats.numstep), cachednumfailedstep(m_stats.numfailedstep) ;

  auto inputstate = state ;
  std::vector<double> stepvector;
  stepvector.reserve(256) ;
  RKErrorCode success = extrapolate(table,state,zout,&jacobian,&stepvector) ;
  if ( success==RKSuccess )
  {
    // now make small changes in tx,ty,qop
    double delta[3] = {0.01,0.01,1e-8} ;
    for(int col=0; col<3; ++col) {
      auto astate = inputstate;
      switch(col) {
      case 0: astate.tx() += delta[0] ; break ;
      case 1: astate.ty() += delta[1] ; break ;
      case 2: astate.qop  += delta[2] ; break ;
      }
      RK::details::Cache<> cache ;
      RK::details::Vec4<> err ;
      boost::apply_visitor( [&](const auto& tbl) {
          for(const auto& step : stepvector) {
            this->evaluateRKStep(tbl.get(),step,astate,err,cache ) ;
            ++cache.step ;
          } }, table );
      if( !(std::abs(state.z - astate.z) < TrackParameters::propagationTolerance ) ) {
        std::cout << "problem in numerical integration. " << std::endl ;
        std::cout << "zin: " << inputstate.z << " "
                  << " zout: " << zout << " "
                  << " state.z: " << state.z << " "
                  << " dstate.z: " << astate.z << std::endl ;
        std::cout << "num step: "
                  << stepvector.size() << " "
                  << m_stats.numstep - cachednumstep << " "
                  << m_stats.numfailedstep - cachednumfailedstep << std::endl ;
      }
      assert(std::abs(state.z - astate.z) < TrackParameters::propagationTolerance ) ;

      for(int row=0; row<4; ++row) {
        jacobian.matrix(row,col) = (astate.parameters(row) - state.parameters(row)) / delta[col] ;
      }
    }
  }
  return success ;
}
