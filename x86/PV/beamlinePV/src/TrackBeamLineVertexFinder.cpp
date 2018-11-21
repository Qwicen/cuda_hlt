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
#include "BeamlinePVConstants.cuh"

#ifdef WITH_ROOT
#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"
#endif
 
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
   
  
  // This implements the adapative vertex fit with Tukey's weights.
  PV::Vertex fitAdaptive( const std::vector<PVTrack>::iterator& tracksbegin,
          const std::vector<PVTrack>::iterator& tracksend,
          const float3& seedposition,
          std::vector<unsigned short>& unusedtracks,
          unsigned short maxNumIter=5,
          float chi2max=9)
  {
    // make vector of TrackInVertex objects
    std::vector<PVTrackInVertex> tracks(tracksbegin,tracksend) ;
    bool converged = false ;

    float3 vtxpos = seedposition;

    PV::Vertex vertex;
    float vtxcov[6];
    vtxcov[0] = 0.;
    vtxcov[1] = 0.;
    vtxcov[2] = 0.;
    vtxcov[3] = 0.;
    vtxcov[4] = 0.;
    vtxcov[5] = 0.;

    const float maxDeltaZConverged{0.001} ;
    float chi2tot = 0;
    unsigned short nselectedtracks = 0;
    unsigned short iter = 0;
    debug_cout << "next vertex " << std::endl;
    for(; iter<maxNumIter && !converged;++iter) {
      PV::myfloat halfD2Chi2DX2_00 = 0.;
      PV::myfloat halfD2Chi2DX2_10 = 0.;
      PV::myfloat halfD2Chi2DX2_11 = 0.;
      PV::myfloat halfD2Chi2DX2_20 = 0.;
      PV::myfloat halfD2Chi2DX2_21 = 0.;
      PV::myfloat halfD2Chi2DX2_22 = 0.;
      float3 halfDChi2DX{0.f,0.f,0.f} ;
      chi2tot = 0.f ;
      nselectedtracks = 0 ;
      float2 vtxposvec{vtxpos.x,vtxpos.y};
      debug_cout << "next track" << std::endl;
      for( auto& trk : tracks ) {
        // compute the chi2
        const float dz = vtxpos.z - trk.z;
        float2 res{0.f,0.f};
        res = vtxposvec - (trk.x + trk.tx*dz);
        
        float chi2 = res.x*res.x * trk.W_00 + res.y*res.y*trk.W_11 ;
        debug_cout << "chi2 = " << chi2 << ", max = " << chi2max << std::endl;
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
          float3 HWr;
          HWr.x = res.x * trk.W_00;
          HWr.y = res.y * trk.W_11;
          HWr.z = -trk.tx.x*res.x*trk.W_00 - trk.tx.y*res.y*trk.W_11;  
                
          halfDChi2DX = halfDChi2DX + HWr * trk.weight;
          
          halfD2Chi2DX2_00 += trk.weight * trk.HWH_00 ;
          halfD2Chi2DX2_10 += 0.f; 
          halfD2Chi2DX2_11 += trk.weight * trk.HWH_11 ;
          halfD2Chi2DX2_20 += trk.weight * trk.HWH_20 ;
          halfD2Chi2DX2_21 += trk.weight * trk.HWH_21 ;
          halfD2Chi2DX2_22 += trk.weight * trk.HWH_22 ;
                    
          chi2tot += trk.weight * chi2 ;
        }
      }
      if(nselectedtracks>=2) {
        // compute the new vertex covariance using analytical inversion
        PV::myfloat a00 = halfD2Chi2DX2_00;
        PV::myfloat a10 = halfD2Chi2DX2_10;
        PV::myfloat a11 = halfD2Chi2DX2_11;
        PV::myfloat a20 = halfD2Chi2DX2_20;
        PV::myfloat a21 = halfD2Chi2DX2_21;
        PV::myfloat a22 = halfD2Chi2DX2_22;
        
        PV::myfloat det = a00 * (a22 * a11 - a21 * a21) - a10 * (a22 * a10 - a21 * a20) + a20 * (a21*a10 - a11*a20);
        // if (det == 0) return false;
                
        vtxcov[0] = (a22*a11 - a21*a21) / det;
        vtxcov[1] = -(a22*a10-a20*a21) / det;
        vtxcov[2] = (a22*a00-a20*a20) / det;
        vtxcov[3] = (a21*a10-a20*a11) / det;
        vtxcov[4] = -(a21*a00-a20*a10) / det;
        vtxcov[5] = (a11*a00-a10*a10) / det;
        
        // compute the delta w.r.t. the reference
        float3 delta{0.f,0.f,0.f};
        // CHECK this
        delta.x = -1.f * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z );
        delta.y = -1.f * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z );
        delta.z = -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z );
        
        // note: this is only correct if chi2 was chi2 of reference!
        chi2tot  += delta.x * halfDChi2DX.x + delta.y * halfDChi2DX.y + delta.z * halfDChi2DX.z;
        
        // update the position
        vtxpos = vtxpos + delta;
        converged = std::abs(delta.z) < maxDeltaZConverged ;
      } else {
        break ;
      }
    } // end iteration loop
    //std::cout << "Number of iterations: " << iter << " " << nselectedtracks << std::endl ;
    vertex.chi2 = chi2tot ;
    vertex.setPosition(vtxpos);
    vertex.setCovMatrix(vtxcov);    
    for( const auto& trk : tracks ) {       
      if( trk.weight > 0 ) 
        vertex.n_tracks++;
      else unusedtracks.push_back( trk.index ) ;     }
    return vertex ;
  }
  
}
 
void findPVs(
  uint* kalmanvelo_states,
  int * velo_atomics,
  uint* velo_track_hit_number,
  PV::Vertex* reconstructed_pvs,
  int* number_of_pvs,
  const uint number_of_events 
) 
{
  
  const int Nbins = (m_zmax-m_zmin)/m_dz ; 
#ifdef WITH_ROOT
  // Histograms only for checking and debugging
  TFile *f = new TFile("../output/PVs.root", "RECREATE");
  //TTree *t_velo_states = new TTree("velo_states", "velo_states");
  TH1F* h_z0[number_of_events];
  TH1F* h_vx[number_of_events];
  TH1F* h_vy[number_of_events];
  TH1F* h_vz[number_of_events];
  for ( int i = 0; i < number_of_events; ++i ) {
    std::string name = "z0_" + std::to_string(i);
    h_z0[i] = new TH1F(name.c_str(), "", Nbins, 0, Nbins-1);
    name = "vx_" + std::to_string(i);
    h_vx[i] = new TH1F(name.c_str(), "", 100, -1, 1);
    name = "vy_" + std::to_string(i);
    h_vy[i] = new TH1F(name.c_str(), "", 100, -1, 1);
    name = "vz_" + std::to_string(i);
    h_vz[i] = new TH1F(name.c_str(), "", 100, -300, 300);
  }
  //t_z0->Branch("z0", &z0, "z0[number_of_events]/F");
#endif
 
  for ( uint event_number = 0; event_number < number_of_events; event_number++ ) {
    debug_cout << "AT EVENT " << event_number << std::endl;
    int &n_pvs = number_of_pvs[event_number];
    n_pvs = 0;

     // get consolidated states
    const Velo::Consolidated::Tracks velo_tracks {(uint*) velo_atomics, velo_track_hit_number, event_number, number_of_events};
    const Velo::Consolidated::States velo_states {kalmanvelo_states, velo_tracks.total_number_of_tracks};
    const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
    const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);
      
    // Step 1: select tracks with velo info, compute the poca to the
    // beamline. cache the covariance matrix at this position. I'd
    // rather us a combination of copy_if and transform, but don't know
    // how to do that efficiently.
    const auto Ntrk = number_of_tracks_event; //tracks.size() ;
    debug_cout << "# of input velo states: " << Ntrk << std::endl;
    std::vector< PVTrack > pvtracks(Ntrk) ; // allocate everything upfront. don't use push_back/emplace_back
    {
      auto it = pvtracks.begin() ;
      for(short unsigned int index = 0; index < Ntrk; ++index) {
        const Velo::State s = velo_states.get(event_tracks_offset + index); 
        // compute the (chance in) z of the poca to the beam axis
        const auto tx = s.tx ;
        const auto ty = s.ty ;
        const double dz = ( tx * ( beamline.x - s.x ) + ty * ( beamline.y - s.y ) ) / (tx*tx+ty*ty) ;
        const double newz = s.z + dz ;
        if( m_zmin < newz  && newz < m_zmax ) {
          *it = PVTrack{s,dz,index} ;
          ++it ;
        }
        
      }
      pvtracks.erase(it,pvtracks.end()) ;
    }
  
    debug_cout << "Selected " << (float)(pvtracks.size() ) / Ntrk << " states for PV seeds " << std::endl;
     
    // Step 2: fill a histogram with the z position of the poca. Use the
    // projected vertex error on that position as the width of a
    // gauss. Divide the gauss properly over the bins. This is quite
    // slow: some simplification may help here.
    
    // we need to define what a bin is: integral between
    //   zmin + ibin*dz and zmin + (ibin+1)*dz
    // we'll have lot's of '0.5' in the code below. at some point we may
    // just want to shift the bins.
    
     // this can be changed into an std::accumulate
  
    std::vector<float> zhisto(Nbins,0.0f) ;
    {
      for( const auto& trk : pvtracks ) {
        // bin in which z0 is, in floating point
        const float zbin = (trk.z - m_zmin)/m_dz ;
      
        // to compute the size of the window, we use the track
        // errors. eventually we can just parametrize this as function of
        // track slope.
        const float zweight = trk.tx.x*trk.tx.x*trk.W_00 + trk.tx.y*trk.tx.y*trk.W_11;
        const float zerr = 1/std::sqrt( zweight );
        // get rid of useless tracks. must be a bit carefull with this.
        if( zerr < m_maxTrackZ0Err) { //m_nsigma < 10*m_dz ) {
          const float halfwindow = GaussApprox::a*zerr / m_dz;
          // this looks a bit funny, but we need the first and last bin of the histogram to remain empty.
          const int minbin = std::max(int( zbin - halfwindow ),1);
          const int maxbin = std::min(int( zbin + halfwindow ),Nbins-2);
          // we can get rid of this if statement if we make a selection of seeds earlier
          if( maxbin >= minbin ) {
            float integral = 0 ;
            for( auto i=minbin; i<maxbin; ++i) {
              const float relz = ( m_zmin + (i+1)*m_dz - trk.z ) /zerr;
              const float thisintegral = GaussApprox::integral( relz );
              zhisto[i] += thisintegral - integral;
              integral = thisintegral;
            }
            // deal with the last bin
            zhisto[maxbin] += 1.f - integral;
          }
        }
      }
    }
#ifdef WITH_ROOT
    for ( int i = 0; i < Nbins; ++i ) {
      h_z0[event_number]->SetBinContent( i, zhisto[i] );
    }
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
        const float threshold = m_dz / (10.f * m_maxTrackZ0Err) ; // need something sensible that depends on binsize
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
      debug_cout << "Found " << clusteredges.size()/2 << " proto clusters" << std::endl;
    
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
    debug_cout << "Found " << clusters.size() << " clusters" << std::endl;
   
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
     auto zClusterMean = [zhisto](auto izmax) -> float {
      const float *b = zhisto.data() + izmax ;
      float d1 = *b - *(b-1) ;
      float d2 = *b - *(b+1) ;
      float idz =  d1+d2>0 ? 0.5f*(d1-d2)/(d1+d2) : 0.0f ;
      return m_zmin + m_dz * (izmax + idz + 0.5f) ;
    };
  
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
    
    for ( auto seed : seedsZWithIteratorPair ) {
      debug_cout << "Associated " << seed.end - seed.begin << " tracks to seed " << std::endl;
    }

    // Step 5: perform the adaptive vertex fit for each seed.
    std::vector<PV::Vertex> vertices ;
    std::vector<unsigned short> unusedtracks ;
    unusedtracks.reserve(pvtracks.size()) ;
 
    for ( const auto seed : seedsZWithIteratorPair ) {
      PV::Vertex vertex = fitAdaptive(seed.begin,seed.end,
                                      float3{beamline.x,beamline.y,seed.z},
                                      unusedtracks,m_maxFitIter,m_maxDeltaChi2) ; 
      vertices.push_back(vertex);
    }

    debug_cout << "Vertices remaining after fitter: " << vertices.size() << std::endl;
    for ( auto vertex : vertices ) {
      debug_cout << "   vertex has " << vertex.n_tracks << " tracks, x = " << vertex.position.x << ", y = " << vertex.position.y << ", z = " << vertex.position.z << std::endl;
    }

    // Steps that we could still take:
    // * remove vertices with too little tracks
    // * assign unused tracks to other vertices
    // * merge vertices that are close

    // create the output container
    const auto maxVertexRho2 = sqr(m_maxVertexRho) ;
    for( const auto& vertex : vertices ) {
      const auto beamlinedx = vertex.position.x - beamline.x ;
      const auto beamlinedy = vertex.position.y - beamline.y ;
      const auto beamlinerho2 = sqr(beamlinedx) + sqr(beamlinedy) ;
#ifdef WITH_ROOT
      h_vx[event_number]->Fill(vertex.position.x);
      h_vy[event_number]->Fill(vertex.position.y);
      h_vz[event_number]->Fill(vertex.position.z);
#endif
      if( vertex.n_tracks >= m_minNumTracksPerVertex && beamlinerho2 < maxVertexRho2 ) {
        reconstructed_pvs[ n_pvs++ ] = vertex;
      }
    }
    
  } // event loop 

#ifdef WITH_ROOT
  f->Write();
  f->Close();
#endif
 
}
