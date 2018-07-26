// Include files

// STL
#include <cmath>
#include <vector>
#include <algorithm>

// local
#include "../include/PVSeedTool.h"


bool  vtxcomp( vtxCluster *first, vtxCluster *second ) {
    return first->z < second->z;
}
bool  multcomp( vtxCluster *first, vtxCluster *second ) {
    return first->ntracks > second->ntracks;
}

// auxiliary class for merging procedure of tracks/clusters
struct pair_to_merge final {

  vtxCluster* first = nullptr;  // pointer to first cluster to be merged
  vtxCluster* second = nullptr; // pointer to second cluster to be merged
  double chi2dist = 10.e+10;    // a chi2dist = zdistance**2/(sigma1**2+sigma2**2)

  pair_to_merge(vtxCluster* f, vtxCluster* s, double chi2) : first(f), second(s), chi2dist(chi2) {}

};

bool paircomp( const pair_to_merge &first, const pair_to_merge &second ) {
  return first.chi2dist < second.chi2dist;

}

//void print_clusters(const std::vector<vtxCluster*>& pvclus) {
//  std::cout << pvclus.size() << " clusters at this iteration" << std::endl;
//  for(const auto& vc : pvclus) {
//     std::cout << vc->ntracks << " " << vc->z << " "
//            <<  vc->sigsq << " " <<  vc->sigsqmin << std::endl;
//  }
//}

constexpr static const int s_p2mstatic = 5000;


//-----------------------------------------------------------------------------
// Implementation file for class : PVSeedTool
//
// 2005-11-19 : Mariusz Witek
//-----------------------------------------------------------------------------

//DECLARE_COMPONENT( PVSeedTool )

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
//PVSeedTool::PVSeedTool() {}

//=============================================================================
// getSeeds
//=============================================================================
int 
PVSeedTool::getSeeds( VeloState * inputTracks,
                     const XYZPoint& beamspot, int number_of_tracks, XYZPoint * seeds)  {
  
  
  //if(inputTracks.size() < 3 ) return seeds;

  vtxCluster  vclusters[number_of_tracks];

  int number_of_clusters = 0;
  for (int i = 0; i < number_of_tracks; i++) {

    double sigsq;
    double zclu;
    auto trk = inputTracks[i];
   
    zclu = zCloseBeam(trk,beamspot);
    errorForPVSeedFinding(trk.tx, trk.ty,sigsq);

    if ( fabs(zclu)>2000.) continue;
    vtxCluster clu;
    clu.z = zclu;
    clu.sigsq = sigsq;
    clu.sigsqmin = clu.sigsq;
    clu.ntracks = 1;
    vclusters[i] = clu;
    number_of_clusters++;
  }



 double  zseeds[m_max_clusters];
  //std::cout << "not broken yet1" << std::endl;
 findClusters(vclusters, zseeds, number_of_clusters);
 std::cout << *(zseeds+1) << std::endl;
 std::cout << "not broken yet2 getClusterCounter()" << getClusterCounter() << std::endl;
  //seeds.reserve(m_max_clusters);
  for(int i = 0; i < getClusterCounter(); i++) {
    //std::cout << i << " not broken yet3 "<< zseeds[i] << std::endl;
    seeds[i] = XYZPoint{ beamspot.x, beamspot.y, zseeds[i]};
  }
  return getClusterCounter();

}


void PVSeedTool::findClusters(vtxCluster * vclus, double * zclusters, int number_of_clusters)  {


  //maybe sort in z before merging?
  
  //why blow up errors??
  
  for(int i = 0; i < number_of_clusters; i++) {
    vclus[i].sigsq *= m_factorToIncreaseErrors*m_factorToIncreaseErrors; // blow up errors
    vclus[i].sigsqmin = vclus[i].sigsq;
  }

  int counter_clusters = 0;
   int counter_merges = -1;
   std::cout<<"inital nubmer of clusters: " << number_of_clusters << std::endl;


   //asume clusters sorted in z
  for(int index_cluster = 0; index_cluster < number_of_clusters; index_cluster++) {
      //only look at next five clusters
    //skip cluster which have already been merged
    
    int second_cluster_counter = 0;
    while(second_cluster_counter < 100) {
      int index_second_cluster = index_cluster +1 + second_cluster_counter;
      if(index_second_cluster >= number_of_clusters) break;
      //skip cluster which have already been merged
      if(vclus[index_cluster].ntracks == 0) break;
      if(vclus[index_second_cluster].ntracks == 0) continue;
      double z1 = vclus[index_cluster].z;
      double z2 = vclus[second_cluster_counter].z;
      double s1 = vclus[index_cluster].sigsq;
      double s2 = vclus[second_cluster_counter].sigsq;
      double s1min = vclus[index_cluster].sigsqmin;
      double s2min = vclus[second_cluster_counter].sigsqmin;
      double sigsqmin = s1min;
      if(s2min<s1min) sigsqmin = s2min;


      double zdist = z1 - z2;
      double chi2dist = zdist*zdist/(s1+s2);
      //merge if chi2dist is smaller than max
      if (chi2dist<m_maxChi2Merge ) {
        double w_inv = (s1*s2/(s1+s2));
        double zmerge = w_inv*(z1/s1+z2/s2);
        std::cout << "before merge: " << vclus[index_cluster].z << " " << vclus[second_cluster_counter].z << " " << chi2dist << " " << zmerge << " " << w_inv<< " " << s1 << " " << s2 << std::endl;

        vclus[index_cluster].z        = zmerge;
        vclus[index_cluster].sigsq    = w_inv;
        vclus[index_cluster].sigsqmin = sigsqmin;
        vclus[index_cluster].ntracks += vclus[second_cluster_counter].ntracks;
        vclus[second_cluster_counter].ntracks  = 0;  // mark second cluster as used
        counter_merges++;
        std::cout << "after merge " << vclus[index_cluster].z << std::endl;
      }
      //stop while loop after first merge
      if (chi2dist<m_maxChi2Merge ) break;

      second_cluster_counter++;
    }
    

   }

   /*
  while(counter_merges != 0) {
    counter_clusters = 0;
     //loop over all clusters and check if something to merge
    //std::cout << "counter merges: "<<counter_merges<<std::endl;
    counter_merges = 0;
    for(int i = 0; i < number_of_clusters ; i++) {
      //don't re-use cluster which have already been merged
      if(vclus[i].ntracks == 0) continue;
      //else counter_clusters++;
      
      for(int j = 0; j < number_of_clusters ; j++) {
        //don't merge with yourself
        if(i==j) continue;
        //don't re-use cluster which have already been merged
        if(vclus[j].ntracks == 0) continue;
        //std::cout << "ij"<<i<<" "<<j<<std::endl;

        double z1 = vclus[i].z;
        double z2 = vclus[j].z;
        double s1 = vclus[i].sigsq;
        double s2 = vclus[j].sigsq;
        double s1min = vclus[i].sigsqmin;
        double s2min = vclus[j].sigsqmin;
        double sigsqmin = s1min;
        if(s2min<s1min) sigsqmin = s2min;


        double zdist = z1 - z2;
        double chi2dist = zdist*zdist/(s1+s2);
        //merge if chi2dist is smaller than max
        if (chi2dist<m_maxChi2Merge ) {
          double w_inv = (s1*s2/(s1+s2));
          double zmerge = w_inv*(z1/s1+z2/s2);
          std::cout << "before merge: " << vclus[i].z << " " << vclus[j].z << " " << chi2dist << " " << zmerge << " " << w_inv<< " " << s1 << " " << s2 << std::endl;

          vclus[i].z        = zmerge;
          vclus[i].sigsq    = w_inv;
          vclus[i].sigsqmin = sigsqmin;
          vclus[i].ntracks += vclus[j].ntracks;
          vclus[j].ntracks  = 0;  // mark second cluster as used
          vclus[i].not_merged = 0;
          vclus[j].not_merged = 0;
          counter_merges++;
          std::cout << "after merge " << vclus[i].z << std::endl;
          //break;



        }
      }
    }
    std::cout << "counter merges: "<<counter_merges << std::endl;
    //count clusters
    for(int i = 0; i < number_of_clusters ; i++) {
      if(vclus[i].ntracks > 1) counter_clusters++;
      if(vclus[i].ntracks > 1)std::cout << vclus[i].z << " " << vclus[i].ntracks << std::endl;
    }
    std::cout << "remaining clusters: " << counter_clusters << std::endl;
  }

    */

  


  resetClusterCounter();
  std::vector<vtxCluster*> pvclus;
  pvclus.reserve(number_of_clusters);

   for(int i = 0; i < counter_clusters; i++) {
    if(vclus[i].ntracks != 0)    pvclus.push_back(&(vclus[i]));
  } 

/*
  for(int i = 0; i < number_of_clusters; i++) {
    vclus[i].sigsq *= m_factorToIncreaseErrors*m_factorToIncreaseErrors; // blow up errors
    vclus[i].sigsqmin = vclus[i].sigsq;
    pvclus.push_back(vclus + i);
  }

  
  std::sort(pvclus.begin(),pvclus.end(),vtxcomp);
  //  print_clusters(pvclus);

  bool more_merging = true;
  while (more_merging) {
  // find pair of clusters for merging

    // refresh flag for this iteration
    for(auto ivc = pvclus.begin(); ivc != pvclus.end(); ivc++) {
      (*ivc)->not_merged = 1;
    }

    // for a given cluster look only up to a few consequtive ones to merge
    // "a few" might become a property?
    auto n_consequtive = std::min(5,static_cast<int>(pvclus.size()));
    auto ivc2up = std::next( pvclus.begin(), n_consequtive);
    std::vector<pair_to_merge> vecp2m; vecp2m.reserve( std::min(static_cast<int>(pvclus.size())*n_consequtive,s_p2mstatic) );
    for(auto ivc1 = pvclus.begin(); ivc1 != pvclus.end()-1; ivc1++) {
      if(ivc2up != pvclus.end()) ++ivc2up;
      for(auto ivc2 = std::next(ivc1); ivc2 != ivc2up; ivc2++) {
        double zdist = (*ivc1)->z-(*ivc2)->z;
        double chi2dist = zdist*zdist/((*ivc1)->sigsq+(*ivc2)->sigsq);
        if(chi2dist<m_maxChi2Merge && vecp2m.size()<s_p2mstatic) {
          vecp2m.emplace_back(*ivc1,*ivc2,chi2dist);
        }
      }
    }
    // done if no more pairs to merge
    if(vecp2m.empty()) {
      more_merging = false;
    } else {
      // sort if number of pairs reasonable. Sorting increases efficency.
      if(vecp2m.size()<100) std::sort(vecp2m.begin(), vecp2m.end(), paircomp);

      // merge pairs
      for(auto itp2m = vecp2m.begin(); itp2m != vecp2m.end(); itp2m++) {
        vtxCluster* pvtx1 = itp2m->first;
        vtxCluster* pvtx2 = itp2m->second;
        if(pvtx1->not_merged == 1 && pvtx2->not_merged == 1) {

          double z1 = pvtx1->z;
          double z2 = pvtx2->z;
          double s1 = pvtx1->sigsq;
          double s2 = pvtx2->sigsq;
          double s1min = pvtx1->sigsqmin;
          double s2min = pvtx2->sigsqmin;
          double sigsqmin = s1min;
          if(s2min<s1min) sigsqmin = s2min;

          double w_inv = (s1*s2/(s1+s2));
          double zmerge = w_inv*(z1/s1+z2/s2);

          pvtx1->z        = zmerge;
          pvtx1->sigsq    = w_inv;
          pvtx1->sigsqmin = sigsqmin;
          pvtx1->ntracks += pvtx2->ntracks;
          pvtx2->ntracks  = 0;  // mark second cluster as used
          pvtx1->not_merged = 0;
          pvtx2->not_merged = 0;
  }
      }

      // remove clusters which where used
      pvclus.erase( std::remove_if( pvclus.begin(), pvclus.end(),
                                    [](const vtxCluster* cl) { return cl->ntracks<1; }),
                    pvclus.end());
    }
  }

  // End of clustering.
*/
  // Sort according to multiplicity

  std::sort(pvclus.begin(),pvclus.end(),multcomp);

  // Select good clusters.

  for(auto ivc=pvclus.begin(); ivc != pvclus.end(); ivc++) {

    int n_tracks_close = 0;
    for(int i = 0; i < number_of_clusters; i++) {
      if(fabs(vclus[i].z - (*ivc)->z ) < m_dzCloseTracksInCluster ) n_tracks_close++; 
    }

    double dist_to_closest = 1000000.;
    if(pvclus.size() > 1) {
      for(auto ivc1=pvclus.begin(); ivc1 != pvclus.end(); ivc1++) {
  if( ivc!=ivc1 && ( fabs((*ivc1)->z-(*ivc)->z) < dist_to_closest) ) {
    dist_to_closest = fabs((*ivc1)->z-(*ivc)->z);
  }
      }
    }

    // ratio to remove clusters made of one low error track and many large error ones
    double rat = (*ivc)->sigsq/(*ivc)->sigsqmin;
    bool igood = false;
    int ntracks = (*ivc)->ntracks;
    if( ntracks >= m_minClusterMult ) {
      if( dist_to_closest>10. && rat<0.95) igood=true;
      if( ntracks >= m_highMult && rat < m_ratioSig2HighMult)  igood=true;
      if( ntracks <  m_highMult && rat < m_ratioSig2LowMult )  igood=true;
    }
    // veto
    if( n_tracks_close < m_minCloseTracksInCluster ) igood = false;
    //std::cout <<igood<< "counter: " << getClusterCounter() << std::endl;
    if(igood)  {zclusters[getClusterCounter()] = ((*ivc)->z); std::cout<< zclusters[getClusterCounter()] <<std::endl; increaseClusterCounter();}

  }

  //  print_clusters(pvclus);
  std::cout << "clustering finishes" << std::endl;
  std::cout << "cpointer: " << std::endl;
  std::cout << "clusters found: " << std::endl;
  for(int index_cluster = 0; index_cluster < number_of_clusters; index_cluster++) {
    if(vclus[index_cluster].ntracks == 0) continue; 
    std::cout << index_cluster << " " << vclus[index_cluster].z << " "<< vclus[index_cluster].sigsq << " " <<  vclus[index_cluster].ntracks<< std::endl;
  
  }
  

}

void PVSeedTool::errorForPVSeedFinding(double tx, double ty, double &sigz2) const {

  // the seeding results depend weakly on this eror parametrization

    double pMean = 3000.; // unit: MeV

    double tanTheta2 =  tx * tx + ty * ty;
    double sinTheta2 =  tanTheta2 / ( 1. + tanTheta2 );

    // assume that first hit in VD at 8 mm
    double distr        = 8.; // unit: mm
    double dist2        = distr*distr/sinTheta2;
    double sigma_ms2    = m_scatCons * m_scatCons * dist2 / (pMean*pMean);
    double fslope2      = 0.0005*0.0005;
    double sigma_slope2 = fslope2*dist2;

    sigz2 = (sigma_ms2 + sigma_slope2) / sinTheta2;
    if(sigz2 == 0) sigz2 = 100.;

}



double PVSeedTool::zCloseBeam( VeloState track, const XYZPoint& beamspot) const {

  XYZPoint tpoint(track.x, track.y, track.z);

  XYZPoint tdir(track.tx, track.ty, 1.);

  double wx = ( 1. + tdir.x * tdir.x ) / track.c00;
  double wy = ( 1. + tdir.y * tdir.y ) / track.c11;

  double x0 = tpoint.x - tpoint.z * tdir.x - beamspot.x;
  double y0 = tpoint.y - tpoint.z * tdir.y - beamspot.y;
  double den = wx * tdir.x * tdir.x + wy * tdir.y * tdir.y;
  double zAtBeam = - ( wx * x0 * tdir.x + wy * y0 * tdir.y ) / den ;

  double xb = tpoint.x + tdir.x * ( zAtBeam - tpoint.z ) - beamspot.x;
  double yb = tpoint.y + tdir.y * ( zAtBeam - tpoint.z ) - beamspot.y;
  double r2AtBeam = xb*xb + yb*yb ;

  return r2AtBeam < 0.5*0.5 ? zAtBeam : 10e8;
}

//=============================================================================
 