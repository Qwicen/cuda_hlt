/*****************************************************************************\
* (c) Copyright 2018 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "TrackBeamLineVertexFinder.h"


/** @class TrackBeamLineVertexFinder TrackBeamLineVertexFinder.cpp
 * 
 * PV finding strategy:
 * step 1: select tracks with velo info and cache some information useful for PV finding
 * step 2: fill a histogram with the z of the poca to the beamline
 * step 3: do a peak search in that histogram ('vertex seeds')
 * step 4: assign tracks to the closest seed ('partitioning')
 * step 5: fit the vertices with an adapative vertex fit
 *
 *  @author Wouter Hulsbergen (Nikhef, 2018)
 **/

class TrackBeamLineVertexFinder
  : public Gaudi::Functional::Transformer<std::vector<LHCb::RecVertex>(const std::vector<LHCb::Track>&)>
{
 public:
  /// Standard constructor
  TrackBeamLineVertexFinder(const std::string& name, ISvcLocator* pSvcLocator);
  /// Execution
  std::vector<LHCb::RecVertex> operator()(const std::vector<LHCb::Track>&) const override;
  /// Initialization
  StatusCode initialize() override ;
private:

#ifdef TIMINGHISTOGRAMMING
  AIDA::IProfile1D* m_timeperstepPr{nullptr} ;
  AIDA::IProfile1D* m_timevsntrksPr{nullptr} ;
  AIDA::IProfile1D* m_timevsnvtxPr{nullptr} ;
#endif
} ;


DECLARE_COMPONENT( TrackBeamLineVertexFinder )

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TrackBeamLineVertexFinder::TrackBeamLineVertexFinder(const std::string& name,
                 ISvcLocator* pSvcLocator) :
Transformer(name , pSvcLocator,
            KeyValue{"InputTracks", LHCb::TrackLocation::Default},
            KeyValue("OutputVertices", LHCb::RecVertexLocation::Primary)) {}

//=============================================================================
// ::initialize()
//=============================================================================
StatusCode TrackBeamLineVertexFinder::initialize()
{
  auto sc = Transformer::initialize();
  m_velodet = getDet<DeVelo>( DeVeloLocation::Default ) ;
#ifdef TIMINGHISTOGRAMMING
  auto hsvc = service<IHistogramSvc>( "HistogramDataSvc", true ) ;
  m_timeperstepPr = hsvc->bookProf(name()+"/timeperstep","time per step",20,-0.5,19.5) ;
  m_timevsntrksPr = hsvc->bookProf(name()+"/timevsntrks","time vs number of tracks",50,-0.5,249.5) ;
  m_timevsnvtxPr = hsvc->bookProf(name()+"/timevsnvtx","time vs number of vertices",12,-0.5,11.5) ;
#endif  
  return sc ;
}

//=============================================================================
// ::execute()
//=============================================================================

namespace {

  namespace GaussApprox
  {
    constexpr int N = 2 ;
    const float a = std::sqrt(double(2*N+3) ) ;
    float integral( float x )
    {
      const float xi = x/a ;
      const float eta = 1 - xi*xi ;
      constexpr float p[] = {0.5,0.25,0.1875,0.15625} ;
      // be careful: if you choose here one order more, you also need to choose 'a' differently (a(N)=sqrt(2N+3))
      return 0.5f + xi * ( p[0] + eta * (p[1] + eta * p[2] ) )  ;
    }
  }
  
  // structure with minimal track info needed for PV search
  struct PVTrack
  {
    PVTrack() {}
    PVTrack( const LHCb::State& state, double dz, unsigned short _index )
      : z{float(state.z()+dz)},
  x{float(state.x()+dz*state.tx()),float(state.y()+dz*state.ty())},
  tx{float(state.tx()),float(state.ty())},index{_index}
    {
      // perhaps we should invert it /before/ switching to single FPP?
      // it doesn't seem to make much difference.
      const auto& V = state.covariance() ;
      auto dz2 = dz*dz ;
      W(0,0) = V(0,0) + 2*dz*V(2,0)        + dz2*V(2,2) ;
      W(1,0) = V(1,0) + dz*(V(3,0)+V(2,1)) + dz2*V(3,2) ;
      W(1,1) = V(1,1) + 2*dz*V(3,1)        + dz2*V(3,3) ;
      W.Invert();
    }
    float z{0} ;
    Gaudi::Vector2F x ;      /// position (x,y)
    Gaudi::Vector2F tx ;     /// direction (tx,ty)
    Gaudi::SymMatrix2x2F W ; /// weightmatrix
    unsigned short index{0} ;/// index in the list with tracks
  } ;
  
  template<typename FTYPE> FTYPE sqr( FTYPE x ) { return x*x ;}

  struct Extremum
  {
    Extremum( unsigned short _index, float _value, float _integral ) :
      index{_index}, value{_value}, integral{_integral} {}
    unsigned short index;
    float value ;
    float integral ;
  } ;
  
  struct Cluster
  {
    Cluster( unsigned short _izfirst, unsigned short _izlast,  unsigned short _izmax ) :
      izfirst{_izfirst}, izlast{_izlast}, izmax{_izmax} {}
    unsigned short izfirst ;
    unsigned short izlast ;
    unsigned short izmax ;
  } ;

  struct SeedZWithIteratorPair
  {
    using iterator = std::vector< PVTrack >::iterator  ;
    float z ;
    iterator begin ;
    iterator end ;
    SeedZWithIteratorPair( float _z, iterator _begin, iterator _end) :
      z{_z},begin{_begin},end{_end} {}
  } ;
  
  // Need a small extension to the track when fitting the
  // vertex. Caching this information doesn't seem to help much
  // though.
  struct PVTrackInVertex : PVTrack
  {
    PVTrackInVertex( const PVTrack& trk )
      : PVTrack{trk}
    {
      ROOT::Math::SMatrix<float,3,2> H;
      H(0,0) = H(1,1) = 1 ;
      H(2,0) = - trk.tx(0) ;
      H(2,1) = - trk.tx(1) ;
      HW  = H*W ;
      HWH = ROOT::Math::Similarity(H,W) ;
    }
    ROOT::Math::SMatrix<float,3,2> HW ;
    Gaudi::SymMatrix3x3F HWH ;
    float weight{1} ;
  } ;

  struct Vertex
  {
    Gaudi::XYZPoint position ;
    Gaudi::SymMatrix3x3 poscov ;
    std::vector<std::pair<unsigned,float> > tracks ; // index to track + weight in vertex fit
    double chi2 ;
  } ;
  
  // This implements the adapative vertex fit with Tukey's weights.
  Vertex fitAdaptive( const std::vector<PVTrack>::iterator& tracksbegin,
          const std::vector<PVTrack>::iterator& tracksend,
          const Gaudi::XYZPoint& seedposition,
          std::vector<unsigned short>& unusedtracks,
          unsigned short maxNumIter=5,
          float chi2max=9)
  {
    // make vector of TrackInVertex objects
    std::vector<PVTrackInVertex> tracks(tracksbegin,tracksend) ;
    bool converged = false ;
    Vertex vertex ;
    auto& vtxpos = vertex.position ;
    auto& vtxcov = vertex.poscov ;
    vtxpos = seedposition;
    const float maxDeltaZConverged{0.001} ;
    double chi2tot{0} ;
    unsigned short nselectedtracks{0} ;
    unsigned short iter{0} ;
    for(; iter<maxNumIter && !converged;++iter) {
      Gaudi::SymMatrix3x3 halfD2Chi2DX2 ;
      Gaudi::Vector3 halfDChi2DX ;
      chi2tot = 0 ;
      nselectedtracks = 0 ;
      Gaudi::Vector2F vtxposvec{float(vtxpos.x()),float(vtxpos.y())} ;
      for( auto& trk : tracks ) {
  // compute the chi2
  const float dz = vtxpos.z() - trk.z ;
  const Gaudi::Vector2F res = vtxposvec - (trk.x + dz*trk.tx) ;
  float chi2 = ROOT::Math::Similarity(res,trk.W) ;
  // compute the weight.
  trk.weight = 0 ;
  if( chi2 < chi2max ) { // to branch or not, that is the question!
    ++nselectedtracks ;
    // Tukey's weight
    trk.weight = sqr( 1.f - chi2 / chi2max ) ;
    //trk.weight = chi2 < 1 ? 1 : sqr( 1. - (chi2-1) / (chi2max-1) ) ;
    // += operator does not work for mixed FP types
    //halfD2Chi2DX2 += trk.weight * trk.HWH ;
    //halfDChi2DX   += trk.weight * trk.HW * res ;
    // if I use expressions, it crashes!
    //const Gaudi::SymMatrix3x3F thisHalfD2Chi2DX2 = weight * ROOT::Math::Similarity(H, trk.W ) ;
    const Gaudi::Vector3F HWr = trk.HW * res ;
    for(int irow=0; irow<3; ++irow) {
      halfDChi2DX(irow) += trk.weight * HWr(irow) ;
      for(int icol=0; icol<=irow; ++icol) 
        halfD2Chi2DX2(irow,icol) += trk.weight * trk.HWH(irow,icol) ;
    }
    chi2tot += trk.weight * chi2 ;
  }
      }
      if(nselectedtracks>=2) {
  // compute the new vertex covariance
  vtxcov = halfD2Chi2DX2 ;
  /*int OK =*/ vtxcov.InvertChol() ;

  // compute the delta w.r.t. the reference
  Gaudi::Vector3 delta = -1.0 * vtxcov * halfDChi2DX ;
  
  // note: this is only correct if chi2 was chi2 of reference!
  chi2tot  += ROOT::Math::Dot(delta,halfDChi2DX) ;

  // update the position
  vtxpos.SetX( vtxpos.x() + delta(0) ) ;
  vtxpos.SetY( vtxpos.y() + delta(1) ) ;
  vtxpos.SetZ( vtxpos.z() + delta(2) ) ;
  converged = std::abs(delta(2)) < maxDeltaZConverged ;
      } else {
  break ;
      }
    } // end iteration loop
    //std::cout << "Number of iterations: " << iter << " " << nselectedtracks << std::endl ;
    vertex.chi2 = chi2tot ;
    vertex.tracks.reserve( tracks.size() ) ;
    for( const auto& trk : tracks ) {
      if( trk.weight > 0 )
  vertex.tracks.emplace_back( trk.index, trk.weight ) ;
      else
  unusedtracks.push_back( trk.index ) ;
    }
    return vertex ;
  }

  // Temporary: class to time the different steps
#ifdef TIMINGHISTOGRAMMING
  class Timer
  {
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::milliseconds nanoseconds;
  public:
    explicit Timer()
      : m_total(0), m_max(0),m_numcalls(0),m_start(high_resolution_clock::now()) {}
    void start() {
      m_start = high_resolution_clock::now();
    }
    double stop()
    {
      auto diff = high_resolution_clock::now() - m_start ;
      double elapsed = std::chrono::duration <double, std::nano> (diff).count()  ;
      m_total += elapsed;
      if( elapsed > m_max) m_max = elapsed ;
      ++m_numcalls ;
      return m_total ;
    }
    double total() const { return m_total ; }
    double average() const { return m_numcalls>0 ? m_total/m_numcalls : 0 ; }
    double maximum() const { return m_max ; }
    size_t numcalls() const { return m_numcalls ; }
  private:
    double m_total ;
    double m_max ;
    size_t m_numcalls ;
    high_resolution_clock::time_point m_start;
  };
#endif
}


std::vector<LHCb::RecVertex> TrackBeamLineVertexFinder::operator()(const std::vector<LHCb::Track>& tracks) const
{
  // Get the beamline. this only accounts for position, not
  // rotation. that's something to improve! I have considered caching
  // this (with a handle for changes in the geometry, but the
  // computation is so fast that it isn't worth it.)
  const auto beamline = 0.5*(Gaudi::XYZVector{m_velodet->halfBoxOffset( DeVelo::LeftHalf)} +
           Gaudi::XYZVector{m_velodet->halfBoxOffset( DeVelo::RightHalf)} ) ;
  
  // get the tracks
#ifdef TIMINGHISTOGRAMMING
  Timer timer[20] ;
  timer[9].start() ;
  timer[1].start() ;
#endif
  // Step 1: select tracks with velo info, compute the poca to the
  // beamline. cache the covariance matrix at this position. I'd
  // rather us a combination of copy_if and transform, but don't know
  // how to do that efficiently.
  const auto Ntrk = tracks.size() ;
  std::vector< PVTrack > pvtracks(Ntrk) ; // allocate everything upfront. don't use push_back/emplace_back
  {
    auto it = pvtracks.begin() ;
    for(short unsigned int index{0}; index<Ntrk; ++index) {
      const auto& trk = tracks[index] ;
      if( trk.hasVelo() ) {
  // compute the (chance in) z of the poca to the beam axis
  const LHCb::State& s = trk.firstState() ;
  const auto tx = s.tx() ;
  const auto ty = s.ty() ;
  const double dz = ( tx * ( beamline.x() - s.x() ) + ty * ( beamline.y() - s.y() ) ) / (tx*tx+ty*ty) ;
  const double newz = s.z() + dz ;
  if( m_zmin < newz  && newz < m_zmax ) {
    *it = PVTrack{s,dz,index} ;
    ++it ;
  }
      }
    }
    pvtracks.erase(it,pvtracks.end()) ;
  }
  
#ifdef TIMINGHISTOGRAMMING
  timer[1].stop() ;
  timer[2].start() ;
#endif
  
    // Step 2: fill a histogram with the z position of the poca. Use the
  // projected vertex error on that position as the width of a
  // gauss. Divide the gauss properly over the bins. This is quite
  // slow: some simplification may help here.

  // we need to define what a bin is: integral between
  //   zmin + ibin*dz and zmin + (ibin+1)*dz
  // we'll have lot's of '0.5' in the code below. at some point we may
  // just want to shift the bins.

  // this can be changed into an std::accumulate
  const int Nbins = (m_zmax-m_zmin)/m_dz ;
  std::vector<float> zhisto(Nbins,0.0f) ;
  {
    for( const auto& trk : pvtracks ) {
      // bin in which z0 is, in floating point
      const float zbin = (trk.z - m_zmin)/m_dz ;
      
      // to compute the size of the window, we use the track
      // errors. eventually we can just parametrize this as function of
      // track slope.
      const float zweight = ROOT::Math::Similarity( trk.W, trk.tx ) ;
      const float zerr = 1/std::sqrt( zweight ) ;
      // get rid of useless tracks. must be a bit carefull with this.
      if( zerr < m_maxTrackZ0Err) { //m_nsigma < 10*m_dz ) {
  const float halfwindow = GaussApprox::a*zerr / m_dz ;
  // this looks a bit funny, but we need the first and last bin of the histogram to remain empty.
  const int minbin = std::max(int( zbin - halfwindow ),1) ;
  const int maxbin = std::min(int( zbin + halfwindow ),Nbins-2) ;
  // we can get rid of this if statement if we make a selection of seeds earlier
  if( maxbin >= minbin ) {
    double integral = 0 ;
    for( auto i=minbin; i<maxbin; ++i) {
      const float relz = ( m_zmin + (i+1)*m_dz - trk.z ) /zerr  ;
      const float thisintegral = GaussApprox::integral( relz ) ;
      zhisto[i] += thisintegral - integral ;
      integral = thisintegral ;
    }
    // deal with the last bin
    zhisto[maxbin] += 1-integral ;
  }
      }
    }
  }

#ifdef TIMINGHISTOGRAMMING  
  timer[2].stop() ;
  timer[3].start() ;
#endif
  
  // Step 3: perform a peak search in the histogram. This used to be
  // very simple but the logic needed to find 'significant dips' made
  // it a bit more complicated. In the end it doesn't matter so much
  // because it takes relatively little time.

  //FIXME: the logic is a bit too complicated here. need to see if we
  //simplify something without loosing efficiency.
  std::vector<Cluster> clusters ;
  {
    // step A: make 'ProtoClusters'
    // Step B: for each such ProtoClusters
    //    - find the significant extrema (an odd number, start with a minimum. you can always achieve this by adding a zero bin at the beginning)
    //      an extremum is a bin-index, plus the integral till that point, plus the content of the bin
    //    - find the highest extremum and
    //       - try and partition at the lowest minimum besides it
    //       - if that doesn't work, try the other extremum
    //       - if that doesn't work, accept as cluster

    // Step A: make 'proto-clusters': these are subsequent bins with non-zero content and an integral above the threshold.
    using BinIndex = unsigned short ;
    std::vector<BinIndex> clusteredges ;
    {
      const float threshold = m_dz / (10*m_maxTrackZ0Err) ; // need something sensible that depends on binsize
      bool prevempty = true ;
      float integral = zhisto[0] ;
      for(BinIndex i=1; i<Nbins; ++i) {
  integral += zhisto[i] ;
  bool empty = zhisto[i] < threshold ;
  if( empty != prevempty ) {
    if( prevempty || integral > m_minTracksInSeed )
      clusteredges.emplace_back( i ) ; // may want to store 'i-1'
    else
      clusteredges.pop_back() ;
    prevempty = empty ;
    integral=0 ;
    //std::cout << "creating cluster edge: "
    //      << i << " " << zhisto[i] << " " << integral << std::endl ;
  }
      }
    }
    // Step B: turn these into clusters. There can be more than one cluster per proto-cluster.
    const size_t Nproto = clusteredges.size()/2 ;
    for(unsigned short i = 0; i< Nproto; ++i) {
      const BinIndex ibegin = clusteredges[i*2] ;
      const BinIndex iend = clusteredges[i*2+1] ;
      //std::cout << "Trying cluster: " << ibegin << " " << iend << std::endl ;
      
      // find the extrema
      const float mindip = m_minDipDensity * m_dz  ; // need to invent something
      const float minpeak = m_minDensity * m_dz  ;

      std::vector<Extremum> extrema ;
      {
  bool rising = true ;
  float integral = zhisto[ibegin] ;
  extrema.emplace_back( ibegin, zhisto[ibegin], integral ) ;
  for(unsigned short i=ibegin; i<iend; ++i) {
    const auto value = zhisto[i] ;
    bool stillrising = zhisto[i+1] > value ;
    if( rising && !stillrising && value >= minpeak ) {
      const auto n = extrema.size() ;
      if( n>=2 ) {
        // check that the previous mimimum was significant. we
        // can still simplify this logic a bit.
        const auto dv1 = extrema[n-2].value - extrema[n-1].value ;
        //const auto di1 = extrema[n-1].index - extrema[n-2].index ;
        const auto dv2 = value - extrema[n-1].value ;
        if( dv1 > mindip && dv2 > mindip )
    extrema.emplace_back( i, value, integral + 0.5f*value ) ;
        else if( dv1 > dv2 )
    extrema.pop_back() ;
        else {
    extrema.pop_back() ;
    extrema.pop_back() ;
    extrema.emplace_back( i, value, integral + 0.5f*value ) ;
        }
      } else {
        extrema.emplace_back( i, value, integral + 0.5f*value ) ;
      }
    } else if( rising != stillrising ) extrema.emplace_back( i, value, integral + 0.5f*value ) ;
    rising = stillrising ;
    integral += value ;
  }
  assert(rising==false) ;
  extrema.emplace_back( iend, zhisto[iend], integral ) ;
      }

      // if( extrema.size() < 3 ) {
      //  warning() << "ERROR: too little extrema found." << extrema.size() << endmsg ;
      //  assert(0) ;
      // }
      // if( extrema.size()%2==0 ) {
      //  warning() << "ERROR: even number of extrema found." << extrema.size() << endmsg ;
      // }
      
      // now partition on  extrema
      const auto N = extrema.size() ;
      std::vector<Cluster> subclusters ;
      if(N>3) {
  for(unsigned int i=1; i<N/2+1; ++i ) {
    if( extrema[2*i].integral - extrema[2*i-2].integral > m_minTracksInSeed ) {
      subclusters.emplace_back( extrema[2*i-2].index, extrema[2*i].index, extrema[2*i-1].index) ;
    }
  }
      }
      if( subclusters.empty() ) {
  //FIXME: still need to get the largest maximum!
  if( extrema[1].value >= minpeak ) 
    clusters.emplace_back( extrema.front().index, extrema.back().index, extrema[1].index ) ;
      } else {
  // adjust the limit of the first and last to extend to the entire protocluster
  subclusters.front().izfirst = ibegin ;
  subclusters.back().izlast = iend ;
  clusters.insert(std::end(clusters),std::begin(subclusters),std::end(subclusters) ) ;
      }
    }
  }

#ifdef TIMINGHISTOGRAMMING  
  timer[3].stop() ;
  timer[4].start() ;
#endif
  
  // Step 4: partition the set of tracks by vertex seed: just
  // choose the closest one. The easiest is to loop over tracks and
  // assign to closest vertex by looping over all vertices. However,
  // that becomes very slow as time is proportional to both tracks and
  // vertices. A better method is to rely on the fact that vertices
  // are sorted in z, and then use std::partition, to partition the
  // track list on the midpoint between two vertices. The logic is
  // slightly complicated to deal with partitions that have too few
  // tracks. I checked it by comparing to the 'slow' method.
  

  // I found that this funny weighted 'maximum' is better than most other inexpensive solutions.
  auto zClusterMean = [this,zhisto](auto izmax) -> float {
    const float *b = zhisto.data() + izmax ;
    float d1 = *b - *(b-1) ;
    float d2 = *b - *(b+1) ;
    float idz =  d1+d2>0 ? 0.5f*(d1-d2)/(d1+d2) : 0.0f ;
    return m_zmin + m_dz * (izmax + idz + 0.5f) ;
  } ;
  
  std::vector<SeedZWithIteratorPair> seedsZWithIteratorPair ;
  seedsZWithIteratorPair.reserve( clusters.size() ) ;
  
  if(!clusters.empty()) {
    std::vector< PVTrack >::iterator it = pvtracks.begin() ;
    int iprev=0 ;
    for( int i=0; i<int(clusters.size())-1; ++i ) {
      //const float zmid = 0.5f*(zseeds[i+1].z+zseeds[i].z) ;
      const float zmid = m_zmin + m_dz * 0.5f* (clusters[i].izlast + clusters[i+1].izfirst + 1.f ) ;
      std::vector< PVTrack >::iterator newit = std::partition( it, pvtracks.end(), [zmid](const auto& trk) { return trk.z < zmid ; } ) ;
      // complicated logic to get rid of partitions that are too small, doign the least amount of work
      if( std::distance( it, newit ) >= m_minNumTracksPerVertex ) {
  seedsZWithIteratorPair.emplace_back( zClusterMean(clusters[i].izmax), it, newit ) ;
  iprev = i ;
      } else {
  // if the partition is too small, then repartition the stuff we
  // have just isolated and assign to the previous and next. You
  // could also 'skip' this partition, but then you do too much
  // work for the next.
  if( !seedsZWithIteratorPair.empty() && newit != it ) {
    const float zmid = m_zmin + m_dz * (clusters[iprev].izlast + clusters[i+1].izfirst+0.5f ) ;
    newit = std::partition( it, newit, [zmid](const auto& trk) { return trk.z < zmid ; } ) ;
    // update the last one
    seedsZWithIteratorPair.back().end = newit ;
  }
      }
      it = newit ;
    }
    // Make sure to add the last partition
    if( std::distance( it, pvtracks.end() ) >= m_minNumTracksPerVertex ) {
      seedsZWithIteratorPair.emplace_back(zClusterMean(clusters.back().izmax) , it, pvtracks.end() ) ;
    } else if( !seedsZWithIteratorPair.empty() ) {
      seedsZWithIteratorPair.back().end = pvtracks.end() ;
    }
  }

#ifdef TIMINGHISTOGRAMMING
  timer[4].stop() ;
  timer[5].start() ;
#endif
  
  // Step 5: perform the adaptive vertex fit for each seed.
  std::vector<Vertex> vertices ;
  std::vector<unsigned short> unusedtracks ;
  unusedtracks.reserve(pvtracks.size()) ;
  std::transform(seedsZWithIteratorPair.begin(),seedsZWithIteratorPair.end(),
     std::back_inserter(vertices),
     [&]( const auto& seed ) {
       return fitAdaptive(seed.begin,seed.end,
              Gaudi::XYZPoint{beamline.x(),beamline.y(),seed.z},
              unusedtracks,m_maxFitIter,m_maxDeltaChi2) ;
     } ) ;

#ifdef TIMINGHISTOGRAMMING  
  timer[5].stop() ;
  timer[6].start() ;
#endif
  
  // Steps that we could still take:
  // * remove vertices with too little tracks
  // * assign unused tracks to other vertices
  // * merge vertices that are close

  // create the output container
  std::vector<LHCb::RecVertex> recvertexcontainer ;
  recvertexcontainer.reserve(vertices.size()) ;
  const auto maxVertexRho2 = sqr(m_maxVertexRho) ;
  for( const auto& vertex : vertices ) {
    const auto beamlinedx = vertex.position.x() - beamline.x() ;
    const auto beamlinedy = vertex.position.y() - beamline.y() ;
    const auto beamlinerho2 = sqr(beamlinedx) + sqr(beamlinedy) ;
    if( vertex.tracks.size()>=m_minNumTracksPerVertex && beamlinerho2 < maxVertexRho2 ) {
      auto& recvertex = recvertexcontainer.emplace_back( vertex.position ) ;
      recvertex.setCovMatrix( vertex.poscov ) ;
      recvertex.setChi2AndDoF( vertex.chi2, 2*vertex.tracks.size()-3 ) ;
      recvertex.setTechnique( LHCb::RecVertex::RecVertexType::Primary ) ;
      for( const auto& dau : vertex.tracks )
  recvertex.addToTracks( &(tracks[ dau.first ]), dau.second ) ;
    }
  }
#ifdef TIMINGHISTOGRAMMING    
  timer[6].stop() ;
  timer[9].stop() ;
  for(int i=0; i<20; ++i)
    m_timeperstepPr->fill(float(i),timer[i].total()) ;
  m_timevsntrksPr->fill(pvtracks.size(), timer[9].total()) ;
  m_timevsnvtxPr->fill(vertices.size(), timer[9].total()) ;
#endif
  return recvertexcontainer ;
}
