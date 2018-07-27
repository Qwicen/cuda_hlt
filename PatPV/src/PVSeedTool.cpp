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




constexpr static const int s_p2mstatic = 5000;



//=============================================================================
// getSeeds
//=============================================================================
int getSeeds( VeloState * inputTracks,
                     const XYZPoint& beamspot, int number_of_tracks, XYZPoint * seeds,  int event_number)  {
  
  
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
    vclusters[number_of_clusters] = clu;
    std::cout << "seed " << number_of_clusters << " " <<  vclusters[number_of_clusters].z << " " << vclusters[number_of_clusters].sigsq << std::endl;
    number_of_clusters++;

  }



 double  zseeds[m_max_clusters];
  //std::cout << "not broken yet1" << std::endl;
 int number_final_clusters = findClusters(vclusters, zseeds, number_of_clusters);
 std::cout << *(zseeds+1) << std::endl;
 std::cout << "not broken yet2 getClusterCounter()" << number_final_clusters << std::endl;
  //seeds.reserve(m_max_clusters);
  for(int i = 0; i < number_final_clusters; i++) {
    //std::cout << i << " not broken yet3 "<< zseeds[i] << std::endl;
    seeds[event_number * PatPV::max_number_vertices + i] = XYZPoint{ beamspot.x, beamspot.y, zseeds[i]};
  }
  return number_final_clusters;

}


int findClusters(vtxCluster * vclus, double * zclusters, int number_of_clusters)  {


  //maybe sort in z before merging?
  
  //why blow up errors??
  
  for(int i = 0; i < number_of_clusters; i++) {
    vclus[i].sigsq *= m_factorToIncreaseErrors*m_factorToIncreaseErrors; // blow up errors
    vclus[i].sigsqmin = vclus[i].sigsq;
  }

  int counter_clusters = 0;
   int counter_merges = -1;
   std::cout<<"inital nubmer of clusters: " << number_of_clusters << std::endl;

  bool no_merges = false;
  while(!no_merges) {
   //asume clusters sorted in z
  no_merges = true;
  for(int index_cluster = 0; index_cluster < number_of_clusters - 1; index_cluster++) {
      //only look at next five clusters
    //skip cluster which have already been merged
    std::cout << "index cluster " << index_cluster << std::endl;
    int second_cluster_counter = 0;
    for(int index_second_cluster = index_cluster + 1; index_second_cluster < number_of_clusters; index_second_cluster++){
      std::cout << second_cluster_counter << std::endl;


      //skip cluster which have already been merged
      if(vclus[index_cluster].ntracks == 0) break;
      if(vclus[index_second_cluster].ntracks == 0) { second_cluster_counter++;continue;}
      double z1 = vclus[index_cluster].z;
      double z2 = vclus[index_second_cluster].z;
      double s1 = vclus[index_cluster].sigsq;
      double s2 = vclus[index_second_cluster].sigsq;
      double s1min = vclus[index_cluster].sigsqmin;
      double s2min = vclus[index_second_cluster].sigsqmin;
      double sigsqmin = s1min;
      if(s2min<s1min) sigsqmin = s2min;


      double zdist = z1 - z2;
      double chi2dist = zdist*zdist/(s1+s2);
      //merge if chi2dist is smaller than max
      std::cout << "current pair before merging: " << vclus[index_cluster].z << " " << vclus[index_cluster].ntracks << " " << vclus[index_second_cluster].ntracks << std::endl;
      if (chi2dist<m_maxChi2Merge ) {
        no_merges = no_merges && false;
        double w_inv = (s1*s2/(s1+s2));
        double zmerge = w_inv*(z1/s1+z2/s2);
        std::cout << "before merge: " << vclus[index_cluster].z << " " << vclus[index_second_cluster].z << " " << chi2dist << " " << zmerge << " " << w_inv<< " " << s1 << " " << s2 << std::endl;

        vclus[index_cluster].z        = zmerge;
        vclus[index_cluster].sigsq    = w_inv;
        vclus[index_cluster].sigsqmin = sigsqmin;
        vclus[index_cluster].ntracks += vclus[index_second_cluster].ntracks;
        vclus[index_second_cluster].ntracks  = 0;  // mark second cluster as used
        counter_merges++;
        std::cout << "after merge " << vclus[index_cluster].z << std::endl;
        break;
      }
      std::cout << "current pair after merging: " << vclus[index_cluster].z << " " << vclus[index_cluster].ntracks << " " << vclus[index_second_cluster].ntracks << std::endl;
      //stop while loop after first merge

      
    }
    

   }
}
 
  


  std::vector<vtxCluster*> pvclus;
  pvclus.reserve(number_of_clusters);
  int return_number_of_clusters = 0;
  for(int i = 0; i < number_of_clusters; i++) {
    if(vclus[i].ntracks != 0)    {zclusters[return_number_of_clusters] = vclus[return_number_of_clusters].z; return_number_of_clusters++;}
  } 

  //still missing: cleaning up the clusters like below
/*
   for(int i = 0; i < number_of_clusters; i++) {
    if(vclus[i].ntracks != 0)    pvclus.push_back(&(vclus[i]));
  } 


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
  
  }*/
  
  return return_number_of_clusters;

}

void errorForPVSeedFinding(double tx, double ty, double &sigz2)  {

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



double zCloseBeam( VeloState track, const XYZPoint& beamspot) {

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
 